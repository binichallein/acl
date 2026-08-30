"""Pure catalog snapshots for the external AppWorld runtime."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import TypeVar, cast

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    _execution_trace_has_canonical_shape,
    _freeze_json_root,
    _semantic_action_has_canonical_shape,
    canonical_json_bytes,
)

_CATALOG_ERROR = "AppWorld catalog is invalid"
_SCHEMA_ERROR = "AppWorld schema is invalid"
_SURFACE_CALL_ERROR = "AppWorld surface call is invalid"
_SEMANTIC_ACTION_ERROR = "AppWorld semantic action is invalid"
_OBSERVATION_ERROR = "AppWorld observation group is invalid"
_TRACE_ERROR = "AppWorld execution trace is invalid"
_INTEGRITY_ERROR = "AppWorld adapter integrity validation failed"
_WITNESS_ERROR = "AppWorld catalog is unprobeable"
_OUTER_KEYS = frozenset({"type", "function"})
_FUNCTION_KEYS = frozenset({"name", "description", "parameters"})
_CALL_KEYS = frozenset({"name", "arguments"})
_REQUESTER_CONTROL_NAMES = frozenset(
    {
        "_api_name",
        "_app_name",
        "_system_datetime",
        "client",
        "raise_on_failure",
        "show",
        "track",
    }
)
_VARIANT_ID = "appworld-source-v1"
_VARIANT_MANIFEST: Mapping[str, JSONValue] = {
    "kind": "appworld_source_interface",
    "schema_policy": "closed-root-v1",
}
_SUPPORTED_DRAFT202012_FORMATS = frozenset(
    {
        "date",
        "date-time",
        "duration",
        "email",
        "hostname",
        "idn-email",
        "idn-hostname",
        "ipv4",
        "ipv6",
        "iri",
        "iri-reference",
        "json-pointer",
        "regex",
        "relative-json-pointer",
        "time",
        "uri",
        "uri-reference",
        "uri-template",
        "uuid",
    }
)
_SINGLE_SUBSCHEMA_KEYS = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SEQUENCE_SUBSCHEMA_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_MAPPING_SUBSCHEMA_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)
_WITNESS_ANNOTATION_KEYS = frozenset(
    {
        "$comment",
        "default",
        "deprecated",
        "description",
        "examples",
        "readOnly",
        "title",
        "writeOnly",
    }
)
_WITNESS_KEYWORDS = _WITNESS_ANNOTATION_KEYS | frozenset(
    {
        "additionalProperties",
        "anyOf",
        "const",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "multipleOf",
        "oneOf",
        "properties",
        "required",
        "type",
    }
)
_WITNESS_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_MAX_WITNESS_SCHEMA_DEPTH = 64
_MAX_WITNESS_STRING_LENGTH = 4096
_MAX_WITNESS_ARRAY_ITEMS = 256
_MAX_WITNESS_NUMERIC_CANDIDATES = 32
_MAX_WITNESS_CANDIDATES = 32
_MAX_WITNESS_VALUE_NODES = 4096
_MAX_WITNESS_CANONICAL_BYTES = 1024 * 1024
_MAX_WITNESS_CANDIDATE_POOL_BYTES = 4 * _MAX_WITNESS_CANONICAL_BYTES
_MAX_SAFE_INTEGER = 2**53 - 1
_APPWORLD_ADAPTER_INTEGRITY_TOKEN = object()
_ResultT = TypeVar("_ResultT")
_FormatRaises = type[Exception] | tuple[type[Exception], ...]
_FormatCheckerEntry = tuple[Callable[[object], bool], _FormatRaises]


class _CatalogSnapshotError(ValueError):
    """Internal marker converted to one payload-free public error."""


class _SchemaSnapshotError(ValueError):
    """Internal marker converted to one payload-free public error."""


class _AdapterIntegrityError(ValueError):
    """Internal marker kept distinct from malformed external values."""


class _WitnessGenerationError(ValueError):
    """Internal marker converted to one payload-free public error."""


@dataclass(frozen=True, slots=True, repr=False)
class _WitnessValueBudget:
    nodes: int
    canonical_bytes: int
    depth: int


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _WitnessCandidateSet:
    raw: tuple[JSONValue, ...]
    valid: tuple[JSONValue, ...]


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _ToolBinding:
    app_name: str
    api_name: str
    schema: Mapping[str, JSONValue]
    schema_canonical: bytes
    validator: Draft202012Validator


@dataclass(frozen=True, slots=True, repr=False)
class _FrozenMappingRootSeal:
    mapping: Mapping[str, JSONValue]
    items: object
    index: object
    index_entries: tuple[tuple[object, object], ...]


@dataclass(frozen=True, slots=True, repr=False)
class _ValidatorSeal:
    validator: Draft202012Validator
    slots: tuple[tuple[str, object], ...]
    validation_entries: tuple[object, ...]
    format_checker: FormatChecker
    checker_mapping: object
    checker_entries: tuple[tuple[object, object], ...]


@dataclass(frozen=True, slots=True, repr=False)
class _ToolRuntimeSeal:
    position: int
    name: str
    app_name: str
    api_name: str
    tool: SurfaceToolSpec
    description: str
    schema: Mapping[str, JSONValue]
    schema_canonical: bytes
    schema_root: _FrozenMappingRootSeal
    binding: _ToolBinding
    validator_seal: _ValidatorSeal


def _snapshot_json(value: object, active: set[int]) -> object:
    if value is None or type(value) in (bool, str, int, float):
        return value

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise _CatalogSnapshotError
        active.add(identity)
        try:
            raw_entries = tuple(value.items())
            entries = tuple(tuple(entry) for entry in raw_entries)
            snapshot: dict[str, object] = {}
            for entry in entries:
                if len(entry) != 2 or type(entry[0]) is not str:
                    raise _CatalogSnapshotError
                key = entry[0]
                if key in snapshot:
                    raise _CatalogSnapshotError
                snapshot[key] = _snapshot_json(entry[1], active)
            return snapshot
        except _CatalogSnapshotError:
            raise
        except Exception:
            raise _CatalogSnapshotError from None
        finally:
            active.remove(identity)

    if type(value) in (list, tuple):
        identity = id(value)
        if identity in active:
            raise _CatalogSnapshotError
        active.add(identity)
        try:
            items = tuple(_snapshot_json(item, active) for item in value)
            return list(items) if type(value) is list else items
        except _CatalogSnapshotError:
            raise
        except Exception:
            raise _CatalogSnapshotError from None
        finally:
            active.remove(identity)

    raise _CatalogSnapshotError


def _snapshot_catalog(
    function_catalog: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], JSONValue, bytes]:
    if isinstance(function_catalog, (str, bytes, bytearray)) or not isinstance(
        function_catalog, Sequence
    ):
        raise ValueError(_CATALOG_ERROR)
    try:
        raw_entries = tuple(function_catalog)
        if not raw_entries:
            raise _CatalogSnapshotError
        snapshots = tuple(_snapshot_json(entry, set()) for entry in raw_entries)
        if any(type(entry) is not dict for entry in snapshots):
            raise _CatalogSnapshotError
        plain = cast(tuple[dict[str, object], ...], snapshots)
        frozen = _freeze_json_root(plain, "AppWorld catalog")
        canonical = canonical_json_bytes(frozen)
        return plain, frozen, canonical
    except _CatalogSnapshotError:
        raise ValueError(_CATALOG_ERROR) from None
    except Exception:
        raise ValueError(_CATALOG_ERROR) from None


def _split_native_name(value: object) -> tuple[str, str]:
    if type(value) is not str:
        raise _CatalogSnapshotError
    app_name, separator, api_name = value.partition("__")
    if (
        separator != "__"
        or not app_name
        or not api_name
        or not app_name.isascii()
        or not api_name.isascii()
        or not app_name.isidentifier()
        or not api_name.isidentifier()
    ):
        raise _CatalogSnapshotError
    return app_name, api_name


def _decode_json_pointer_token(value: str) -> str:
    decoded: list[str] = []
    position = 0
    while position < len(value):
        character = value[position]
        if character != "~":
            decoded.append(character)
            position += 1
            continue
        if position + 1 >= len(value) or value[position + 1] not in {"0", "1"}:
            raise _SchemaSnapshotError
        decoded.append("~" if value[position + 1] == "0" else "/")
        position += 2
    return "".join(decoded)


def _resolve_local_json_pointer(root: object, reference: object) -> object:
    if type(reference) is not str or (
        reference != "#" and not reference.startswith("#/")
    ) or "%" in reference:
        raise _SchemaSnapshotError
    current = root
    if reference == "#":
        return current
    for encoded_token in reference[2:].split("/"):
        pointer_segment = _decode_json_pointer_token(encoded_token)
        if type(current) is dict:
            if pointer_segment not in current:
                raise _SchemaSnapshotError
            current = current[pointer_segment]
        elif type(current) is list:
            if (
                not pointer_segment.isascii()
                or not pointer_segment.isdecimal()
                or (len(pointer_segment) > 1 and pointer_segment.startswith("0"))
            ):
                raise _SchemaSnapshotError
            index = int(pointer_segment)
            if index >= len(current):
                raise _SchemaSnapshotError
            current = current[index]
        else:
            raise _SchemaSnapshotError
    return current


def _schema_format_entries(
    schema: dict[str, object],
) -> tuple[tuple[str, _FormatCheckerEntry], ...]:

    def visit(node: object, active: set[int]) -> None:
        if type(node) is bool:
            return
        if type(node) is not dict:
            raise _SchemaSnapshotError
        identity = id(node)
        if identity in active:
            raise _SchemaSnapshotError
        active.add(identity)
        try:
            if "$schema" in node:
                raise _SchemaSnapshotError
            if "format" in node:
                format_name = node["format"]
                if (
                    type(format_name) is not str
                    or format_name not in _SUPPORTED_DRAFT202012_FORMATS
                ):
                    raise _SchemaSnapshotError
            if "$id" in node or "id" in node:
                raise _SchemaSnapshotError
            if "$dynamicRef" in node or "$recursiveRef" in node:
                raise _SchemaSnapshotError
            if "$ref" in node:
                target = _resolve_local_json_pointer(schema, node["$ref"])
                if type(target) not in (bool, dict):
                    raise _SchemaSnapshotError
                if type(target) is dict:
                    try:
                        Draft202012Validator.check_schema(target)
                    except Exception:
                        raise _SchemaSnapshotError from None
                visit(target, active)

            for keyword in _SINGLE_SUBSCHEMA_KEYS:
                if keyword in node:
                    visit(node[keyword], active)

            for keyword in _SEQUENCE_SUBSCHEMA_KEYS:
                if keyword not in node:
                    continue
                children = node[keyword]
                if type(children) is not list:
                    raise _SchemaSnapshotError
                for child in children:
                    visit(child, active)

            for keyword in _MAPPING_SUBSCHEMA_KEYS:
                if keyword not in node:
                    continue
                children = node[keyword]
                if type(children) is not dict:
                    raise _SchemaSnapshotError
                for child in children.values():
                    visit(child, active)

            if "dependencies" in node:
                dependencies = node["dependencies"]
                if type(dependencies) is not dict:
                    raise _SchemaSnapshotError
                for dependency in dependencies.values():
                    if type(dependency) in (bool, dict):
                        visit(dependency, active)
                    elif type(dependency) is not list:
                        raise _SchemaSnapshotError
        finally:
            active.remove(identity)

    visit(schema, set())
    try:
        registry = Draft202012Validator.FORMAT_CHECKER.checkers
        if type(registry) is not dict:
            raise _SchemaSnapshotError
        entries: list[tuple[str, _FormatCheckerEntry]] = []
        for format_name in sorted(_SUPPORTED_DRAFT202012_FORMATS):
            entry = registry.get(format_name)
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or not callable(entry[0])
            ):
                raise _SchemaSnapshotError
            entries.append((format_name, cast(_FormatCheckerEntry, entry)))
        return tuple(entries)
    except _SchemaSnapshotError:
        raise
    except Exception:
        raise _SchemaSnapshotError from None


def _new_format_checker(
    entries: tuple[tuple[str, _FormatCheckerEntry], ...],
) -> FormatChecker:
    checker = FormatChecker(formats=())
    checker.checkers.update(dict(entries))
    return checker


def _closed_schema(
    value: object,
) -> tuple[dict[str, object], tuple[tuple[str, _FormatCheckerEntry], ...]]:
    try:
        if type(value) is not dict:
            raise _SchemaSnapshotError
        schema = dict(value)
        if schema.get("type") != "object":
            raise _SchemaSnapshotError
        properties = schema.get("properties")
        if type(properties) is not dict:
            raise _SchemaSnapshotError
        if "patternProperties" in schema:
            raise _SchemaSnapshotError
        if _REQUESTER_CONTROL_NAMES.intersection(properties):
            raise _SchemaSnapshotError
        required = schema.get("required")
        if type(required) is not list:
            raise _SchemaSnapshotError
        if (
            any(type(name) is not str for name in required)
            or len(required) != len(set(required))
            or any(name not in properties for name in required)
        ):
            raise _SchemaSnapshotError
        if "additionalProperties" in schema:
            if schema["additionalProperties"] is not False:
                raise _SchemaSnapshotError
        else:
            schema["additionalProperties"] = False
        Draft202012Validator.check_schema(schema)
        return schema, _schema_format_entries(schema)
    except _SchemaSnapshotError:
        raise ValueError(_SCHEMA_ERROR) from None
    except Exception:
        raise ValueError(_SCHEMA_ERROR) from None


def _catalog_records(
    entries: tuple[dict[str, object], ...],
) -> tuple[
    tuple[
        str,
        str,
        str,
        str,
        dict[str, object],
        tuple[tuple[str, _FormatCheckerEntry], ...],
    ],
    ...,
]:
    records: list[
        tuple[
            str,
            str,
            str,
            str,
            dict[str, object],
            tuple[tuple[str, _FormatCheckerEntry], ...],
        ]
    ] = []
    names: set[str] = set()
    try:
        for entry in entries:
            if set(entry) != _OUTER_KEYS or entry.get("type") != "function":
                raise _CatalogSnapshotError
            function = entry.get("function")
            if type(function) is not dict or set(function) != _FUNCTION_KEYS:
                raise _CatalogSnapshotError
            name = function.get("name")
            app_name, api_name = _split_native_name(name)
            if cast(str, name) in names:
                raise _CatalogSnapshotError
            names.add(cast(str, name))
            description = function.get("description")
            if type(description) is not str:
                raise _CatalogSnapshotError
            description.encode("utf-8")
            schema, format_entries = _closed_schema(function.get("parameters"))
            records.append(
                (
                    cast(str, name),
                    app_name,
                    api_name,
                    description,
                    schema,
                    format_entries,
                )
            )
        return tuple(sorted(records, key=lambda record: record[0]))
    except _CatalogSnapshotError:
        raise ValueError(_CATALOG_ERROR) from None
    except UnicodeEncodeError:
        raise ValueError(_CATALOG_ERROR) from None


def _make_frozen_mapping_root_seal(
    mapping: Mapping[str, JSONValue],
) -> _FrozenMappingRootSeal:
    try:
        items = object.__getattribute__(mapping, "_items")
        index = object.__getattribute__(mapping, "_index")
        if type(items) is not tuple or type(index) is not dict:
            raise ValueError
        index_entries = tuple(dict.items(index))
        return _FrozenMappingRootSeal(mapping, items, index, index_entries)
    except Exception:
        raise ValueError(_CATALOG_ERROR) from None


def _frozen_mapping_root_matches(seal: _FrozenMappingRootSeal) -> bool:
    try:
        return (
            object.__getattribute__(seal.mapping, "_items") is seal.items
            and object.__getattribute__(seal.mapping, "_index") is seal.index
            and tuple(dict.items(cast(dict[object, object], seal.index)))
            == seal.index_entries
        )
    except Exception:
        return False


def _make_validator_seal(validator: Draft202012Validator) -> _ValidatorSeal:
    try:
        slot_names = tuple(
            name for name in type(validator).__slots__ if name != "__weakref__"
        )
        slots = tuple(
            (name, object.__getattribute__(validator, name)) for name in slot_names
        )
        validation_plan = object.__getattribute__(validator, "_validators")
        if type(validation_plan) is not list:
            raise ValueError
        validation_entries = tuple(validation_plan)
        format_checker = object.__getattribute__(validator, "format_checker")
        if type(format_checker) is not FormatChecker:
            raise ValueError
        checker_mapping = object.__getattribute__(format_checker, "checkers")
        if type(checker_mapping) is not dict:
            raise ValueError
        checker_entries = tuple(dict.items(checker_mapping))
        return _ValidatorSeal(
            validator,
            slots,
            validation_entries,
            format_checker,
            checker_mapping,
            checker_entries,
        )
    except Exception:
        raise ValueError(_CATALOG_ERROR) from None


def _validator_seal_matches(seal: _ValidatorSeal) -> bool:
    try:
        validator = seal.validator
        if type(validator) is not Draft202012Validator:
            return False
        for name, expected in seal.slots:
            if object.__getattribute__(validator, name) is not expected:
                return False
        validation_plan = object.__getattribute__(validator, "_validators")
        if type(validation_plan) is not list or len(validation_plan) != len(
            seal.validation_entries
        ):
            return False
        if any(
            current is not expected
            for current, expected in zip(
                validation_plan,
                seal.validation_entries,
                strict=True,
            )
        ):
            return False
        format_checker = object.__getattribute__(validator, "format_checker")
        if format_checker is not seal.format_checker:
            return False
        checker_mapping = object.__getattribute__(format_checker, "checkers")
        if checker_mapping is not seal.checker_mapping or type(checker_mapping) is not dict:
            return False
        current_entries = tuple(dict.items(checker_mapping))
        return len(current_entries) == len(seal.checker_entries) and all(
            current_key == expected_key and current_value is expected_value
            for (current_key, current_value), (expected_key, expected_value) in zip(
                current_entries,
                seal.checker_entries,
                strict=True,
            )
        )
    except Exception:
        return False


def _snapshot_runtime_json(value: object, error_message: str) -> JSONValue:
    try:
        detached = _snapshot_json(value, set())
        return _freeze_json_root(detached, "AppWorld runtime value")
    except Exception:
        raise ValueError(error_message) from None


def _snapshot_runtime_mapping(
    value: object,
    error_message: str,
) -> Mapping[str, JSONValue]:
    snapshot = _snapshot_runtime_json(value, error_message)
    if not isinstance(snapshot, Mapping):
        raise ValueError(error_message)
    return cast(Mapping[str, JSONValue], snapshot)


def _plain_json(value: JSONValue, error_message: str) -> object:
    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except Exception:
        raise ValueError(error_message) from None


def _action_snapshot(
    action: object,
    error_message: str,
) -> SemanticAction:
    try:
        if not _semantic_action_has_canonical_shape(action):
            raise ValueError
        typed = cast(SemanticAction, action)
        name = object.__getattribute__(typed, "name")
        arguments = _snapshot_runtime_mapping(
            object.__getattribute__(typed, "arguments"),
            error_message,
        )
        rebuilt = SemanticAction(name, arguments)
        stored_canonical = object.__getattribute__(typed, "_arguments_canonical")
        rebuilt_canonical = object.__getattribute__(rebuilt, "_arguments_canonical")
        if stored_canonical != rebuilt_canonical:
            raise ValueError
        return rebuilt
    except _AdapterIntegrityError:
        raise
    except Exception:
        raise ValueError(error_message) from None


def _actions_equal(left: SemanticAction, right: SemanticAction) -> bool:
    try:
        return (
            object.__getattribute__(left, "name")
            == object.__getattribute__(right, "name")
            and object.__getattribute__(left, "_arguments_canonical")
            == object.__getattribute__(right, "_arguments_canonical")
        )
    except Exception:
        return False


def _witness_effective_type(schema: dict[str, object]) -> str | None:
    type_value = schema.get("type")
    if type_value is None:
        return None
    if type(type_value) is str and type_value in _WITNESS_TYPES:
        return type_value
    if (
        type(type_value) is list
        and len(type_value) == 2
        and all(type(item) is str for item in type_value)
        and len(set(type_value)) == 2
        and "null" in type_value
    ):
        non_null = next(item for item in type_value if item != "null")
        if non_null in _WITNESS_TYPES - {"null"}:
            return cast(str, non_null)
    raise _WitnessGenerationError


def _canonical_scalar_size(value: bool | int | float | str | None) -> int:
    try:
        return len(
            json.dumps(
                value,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    except Exception:
        raise _WitnessGenerationError from None


def _require_witness_value_budget(budget: _WitnessValueBudget) -> None:
    if (
        budget.nodes > _MAX_WITNESS_VALUE_NODES
        or budget.canonical_bytes > _MAX_WITNESS_CANONICAL_BYTES
        or budget.depth > _MAX_WITNESS_SCHEMA_DEPTH
    ):
        raise _WitnessGenerationError


def _audit_witness_value_budget(value: object) -> _WitnessValueBudget:
    def visit(
        item: object,
        container_depth: int,
        active: set[int],
    ) -> _WitnessValueBudget:
        if container_depth > _MAX_WITNESS_SCHEMA_DEPTH:
            raise _WitnessGenerationError
        if item is None or type(item) is bool:
            budget = _WitnessValueBudget(1, _canonical_scalar_size(item), 0)
            _require_witness_value_budget(budget)
            return budget
        if type(item) is str:
            if len(item) > _MAX_WITNESS_STRING_LENGTH:
                raise _WitnessGenerationError
            budget = _WitnessValueBudget(1, _canonical_scalar_size(item), 0)
            _require_witness_value_budget(budget)
            return budget
        if type(item) is int:
            if not -_MAX_SAFE_INTEGER <= item <= _MAX_SAFE_INTEGER:
                raise _WitnessGenerationError
            budget = _WitnessValueBudget(1, _canonical_scalar_size(item), 0)
            _require_witness_value_budget(budget)
            return budget
        if type(item) is float:
            if not math.isfinite(item):
                raise _WitnessGenerationError
            budget = _WitnessValueBudget(1, _canonical_scalar_size(item), 0)
            _require_witness_value_budget(budget)
            return budget
        if type(item) not in (list, dict):
            raise _WitnessGenerationError

        identity = id(item)
        if identity in active:
            raise _WitnessGenerationError
        active.add(identity)
        try:
            if len(item) > _MAX_WITNESS_ARRAY_ITEMS:
                raise _WitnessGenerationError
            nodes = 1
            canonical_bytes = 2
            maximum_child_depth = 0
            entries = (
                ((None, child) for child in item)
                if type(item) is list
                else item.items()
            )
            for position, (key, child) in enumerate(entries):
                if position:
                    canonical_bytes += 1
                if type(item) is dict:
                    if type(key) is not str or len(key) > _MAX_WITNESS_STRING_LENGTH:
                        raise _WitnessGenerationError
                    canonical_bytes += _canonical_scalar_size(key) + 1
                child_budget = visit(child, container_depth + 1, active)
                nodes += child_budget.nodes
                canonical_bytes += child_budget.canonical_bytes
                maximum_child_depth = max(maximum_child_depth, child_budget.depth)
                _require_witness_value_budget(
                    _WitnessValueBudget(
                        nodes,
                        canonical_bytes,
                        maximum_child_depth + 1,
                    )
                )
            budget = _WitnessValueBudget(
                nodes,
                canonical_bytes,
                maximum_child_depth + 1,
            )
            _require_witness_value_budget(budget)
            return budget
        finally:
            active.remove(identity)

    return visit(value, 0, set())


def _project_uniform_array_budget(item: object, length: int) -> _WitnessValueBudget:
    if length == 0:
        budget = _WitnessValueBudget(1, 2, 1)
        _require_witness_value_budget(budget)
        return budget
    item_budget = _audit_witness_value_budget(item)
    budget = _WitnessValueBudget(
        1 + length * item_budget.nodes,
        2 + length * item_budget.canonical_bytes + max(0, length - 1),
        item_budget.depth + 1,
    )
    _require_witness_value_budget(budget)
    return budget


def _materialize_uniform_array(item: JSONValue, length: int) -> list[JSONValue]:
    return [item for _ in range(length)]


def _project_object_candidate_budget(
    values: Mapping[str, JSONValue],
    name: str,
    value: JSONValue,
) -> _WitnessValueBudget:
    if type(name) is not str or len(name) > _MAX_WITNESS_STRING_LENGTH:
        raise _WitnessGenerationError
    projected_length = len(values) + int(name not in values)
    if projected_length > _MAX_WITNESS_ARRAY_ITEMS:
        raise _WitnessGenerationError

    nodes = 1
    canonical_bytes = 2
    maximum_child_depth = 0
    position = 0
    found = False
    for key, current in values.items():
        if type(key) is not str or len(key) > _MAX_WITNESS_STRING_LENGTH:
            raise _WitnessGenerationError
        if position:
            canonical_bytes += 1
        if key == name:
            current = value
            found = True
        child_budget = _audit_witness_value_budget(current)
        nodes += child_budget.nodes
        canonical_bytes += _canonical_scalar_size(key) + 1
        canonical_bytes += child_budget.canonical_bytes
        maximum_child_depth = max(maximum_child_depth, child_budget.depth)
        _require_witness_value_budget(
            _WitnessValueBudget(
                nodes,
                canonical_bytes,
                maximum_child_depth + 1,
            )
        )
        position += 1

    if not found:
        if position:
            canonical_bytes += 1
        child_budget = _audit_witness_value_budget(value)
        nodes += child_budget.nodes
        canonical_bytes += _canonical_scalar_size(name) + 1
        canonical_bytes += child_budget.canonical_bytes
        maximum_child_depth = max(maximum_child_depth, child_budget.depth)

    budget = _WitnessValueBudget(
        nodes,
        canonical_bytes,
        maximum_child_depth + 1,
    )
    _require_witness_value_budget(budget)
    return budget


def _materialize_object_candidate(
    values: Mapping[str, JSONValue],
    name: str,
    value: JSONValue,
) -> dict[str, JSONValue]:
    candidate = dict(values)
    candidate[name] = value
    return candidate


def _audit_witness_schema(node: object, depth: int = 0) -> None:
    if depth > _MAX_WITNESS_SCHEMA_DEPTH:
        raise _WitnessGenerationError
    if type(node) is bool:
        return
    if type(node) is not dict or not set(node).issubset(_WITNESS_KEYWORDS):
        raise _WitnessGenerationError

    combinators = tuple(key for key in ("anyOf", "oneOf") if key in node)
    if combinators:
        if len(combinators) != 1 or not set(node).issubset(
            _WITNESS_ANNOTATION_KEYS | {combinators[0]}
        ):
            raise _WitnessGenerationError
        branches = node[combinators[0]]
        if (
            type(branches) is not list
            or not branches
            or len(branches) > _MAX_WITNESS_CANDIDATES
        ):
            raise _WitnessGenerationError
        for branch in branches:
            _audit_witness_schema(branch, depth + 1)
        return

    effective_type = _witness_effective_type(node)
    common_keys = _WITNESS_ANNOTATION_KEYS | {"const", "enum", "type"}
    type_keys: dict[str, frozenset[str]] = {
        "array": frozenset({"items", "minItems", "maxItems"}),
        "boolean": frozenset(),
        "integer": frozenset(
            {
                "exclusiveMaximum",
                "exclusiveMinimum",
                "maximum",
                "minimum",
                "multipleOf",
            }
        ),
        "null": frozenset(),
        "number": frozenset(
            {
                "exclusiveMaximum",
                "exclusiveMinimum",
                "maximum",
                "minimum",
                "multipleOf",
            }
        ),
        "object": frozenset({"additionalProperties", "properties", "required"}),
        "string": frozenset({"format", "maxLength", "minLength"}),
    }
    allowed = common_keys if effective_type is None else common_keys | type_keys[effective_type]
    if not set(node).issubset(allowed):
        raise _WitnessGenerationError
    if effective_type is None and not ({"const", "enum"} & set(node)) and (
        set(node) - _WITNESS_ANNOTATION_KEYS
    ):
        raise _WitnessGenerationError

    if "enum" in node:
        enum_values = node["enum"]
        if (
            type(enum_values) is not list
            or not enum_values
            or len(enum_values) > _MAX_WITNESS_CANDIDATES
        ):
            raise _WitnessGenerationError
        for enum_value in enum_values:
            _audit_witness_value_budget(enum_value)
    if "const" in node:
        _audit_witness_value_budget(node["const"])

    if effective_type == "string":
        minimum_length = node.get("minLength", 0)
        maximum_length = node.get("maxLength")
        if (
            type(minimum_length) is not int
            or minimum_length < 0
            or minimum_length > _MAX_WITNESS_STRING_LENGTH
            or (
                maximum_length is not None
                and (type(maximum_length) is not int or maximum_length < 0)
            )
        ):
            raise _WitnessGenerationError
        format_name = node.get("format")
        if format_name is not None and (
            type(format_name) is not str
            or format_name not in _SUPPORTED_DRAFT202012_FORMATS
        ):
            raise _WitnessGenerationError

    if effective_type == "array":
        minimum_items = node.get("minItems", 0)
        maximum_items = node.get("maxItems")
        if (
            type(minimum_items) is not int
            or minimum_items < 0
            or minimum_items > _MAX_WITNESS_ARRAY_ITEMS
            or (
                maximum_items is not None
                and (type(maximum_items) is not int or maximum_items < 0)
            )
        ):
            raise _WitnessGenerationError
        if "items" in node:
            _audit_witness_schema(node["items"], depth + 1)

    if effective_type == "object":
        properties = node.get("properties", {})
        required = node.get("required", [])
        additional = node.get("additionalProperties", True)
        if (
            type(properties) is not dict
            or len(properties) > _MAX_WITNESS_ARRAY_ITEMS
            or type(required) is not list
            or len(required) > _MAX_WITNESS_ARRAY_ITEMS
            or any(type(name) is not str or name not in properties for name in required)
            or len(required) != len(set(required))
            or type(additional) is not bool
        ):
            raise _WitnessGenerationError
        for property_schema in properties.values():
            _audit_witness_schema(property_schema, depth + 1)

    if effective_type in {"integer", "number"}:
        for keyword in (
            "exclusiveMaximum",
            "exclusiveMinimum",
            "maximum",
            "minimum",
            "multipleOf",
        ):
            if keyword in node and (
                type(node[keyword]) not in (int, float)
                or not math.isfinite(node[keyword])
            ):
                raise _WitnessGenerationError
        if "multipleOf" in node and cast(int | float, node["multipleOf"]) <= 0:
            raise _WitnessGenerationError
        if effective_type == "integer" and "multipleOf" in node and type(
            node["multipleOf"]
        ) is not int:
            raise _WitnessGenerationError


def _plain_witness_value(value: object) -> JSONValue:
    try:
        _audit_witness_value_budget(value)
        frozen = _snapshot_runtime_json(value, _WITNESS_ERROR)
        plain = cast(JSONValue, _plain_json(frozen, _WITNESS_ERROR))
        _audit_witness_value_budget(plain)
        return plain
    except Exception:
        raise _WitnessGenerationError from None


def _candidate_preference(value: JSONValue) -> tuple[object, ...]:
    if value is None:
        return (0,)
    if type(value) is bool:
        return (1, int(value))
    if type(value) in (int, float):
        numeric = cast(int | float, value)
        return (2, abs(numeric), numeric, type(value).__name__)
    if type(value) is str:
        return (3, len(value), value)
    if type(value) is list:
        return (
            4,
            len(value),
            tuple(_candidate_preference(item) for item in value),
        )
    if type(value) is dict:
        return (
            5,
            len(value),
            tuple(
                (key, _candidate_preference(cast(JSONValue, item)))
                for key, item in sorted(value.items())
            ),
        )
    raise _WitnessGenerationError


def _candidate_sort_key(value: object) -> tuple[object, ...]:
    try:
        plain = _plain_witness_value(value)
        return (*_candidate_preference(plain), canonical_json_bytes(plain))
    except _WitnessGenerationError:
        raise
    except Exception:
        raise _WitnessGenerationError from None


def _bounded_distinct_candidates(
    values: Sequence[object],
    *,
    limit: int = _MAX_WITNESS_CANDIDATES,
) -> tuple[JSONValue, ...]:
    try:
        distinct: dict[bytes, JSONValue] = {}
        aggregate_bytes = 0
        for value in values:
            plain = _plain_witness_value(value)
            canonical = canonical_json_bytes(plain)
            if canonical not in distinct:
                aggregate_bytes += len(canonical)
                if aggregate_bytes > _MAX_WITNESS_CANDIDATE_POOL_BYTES:
                    raise _WitnessGenerationError
                distinct[canonical] = plain
            if len(distinct) > limit:
                raise _WitnessGenerationError
        return tuple(sorted(distinct.values(), key=_candidate_sort_key))
    except _WitnessGenerationError:
        raise
    except Exception:
        raise _WitnessGenerationError from None


def _ascii_hostname_candidate(length: int) -> str | None:
    if length < 1 or length > 253:
        return None
    labels: list[str] = []
    remaining = length
    while remaining > 63:
        label_length = min(63, remaining - 2)
        if label_length < 1:
            return None
        labels.append("a" * label_length)
        remaining -= label_length + 1
    labels.append("a" * remaining)
    return ".".join(labels)


def _string_format_candidates(
    format_name: str,
    minimum_length: int,
    maximum_length: int | None,
) -> tuple[str, ...]:
    base: dict[str, str] = {
        "date": "2000-01-01",
        "date-time": "2000-01-01T00:00:00Z",
        "duration": "P0D",
        "email": "a@example.com",
        "hostname": "a",
        "idn-email": "a@example.com",
        "idn-hostname": "a",
        "ipv4": "0.0.0.0",
        "ipv6": "::",
        "iri": "https://example.com/",
        "iri-reference": "",
        "json-pointer": "",
        "regex": "",
        "relative-json-pointer": "0",
        "time": "00:00:00Z",
        "uri": "https://example.com/",
        "uri-reference": "",
        "uri-template": "",
        "uuid": "00000000-0000-0000-0000-000000000000",
    }
    if format_name not in base:
        raise _WitnessGenerationError
    values = [base[format_name]]
    if format_name in {"email", "idn-email"}:
        values.append("@")
    elif format_name in {"iri", "uri"}:
        values.append("a:b")
    length_hints: set[int] = set()
    if not any(
        len(value) >= minimum_length
        and (maximum_length is None or len(value) <= maximum_length)
        for value in values
    ):
        length_hints.add(minimum_length)
        if maximum_length is not None and maximum_length <= _MAX_WITNESS_STRING_LENGTH:
            length_hints.add(maximum_length)
    if maximum_length is not None and maximum_length < _MAX_WITNESS_STRING_LENGTH:
        length_hints.add(maximum_length + 1)
    for desired in sorted(set(length_hints)):
        if desired < 0 or desired > _MAX_WITNESS_STRING_LENGTH:
            raise _WitnessGenerationError
        if format_name in {"email", "idn-email"}:
            if desired >= 3:
                local_length = min(64, desired - 2)
                domain = _ascii_hostname_candidate(desired - local_length - 1)
                if domain is not None:
                    values.append("a" * local_length + "@" + domain)
        elif format_name in {"hostname", "idn-hostname"}:
            hostname = _ascii_hostname_candidate(desired)
            if hostname is not None:
                values.append(hostname)
        elif format_name in {"iri", "uri"} and desired >= 3:
            values.append("a:" + "b" * (desired - 2))
        elif format_name in {"iri-reference", "uri-reference", "uri-template", "regex"}:
            values.append("a" * desired)
        elif format_name == "json-pointer":
            values.append("" if desired == 0 else "/" + "a" * (desired - 1))
        elif format_name == "relative-json-pointer" and desired >= 1:
            values.append("0" if desired == 1 else "0/" + "a" * (desired - 2))
        elif format_name == "date-time" and desired >= 22:
            values.append(
                "2000-01-01T00:00:00." + "0" * (desired - 21) + "Z"
            )
        elif format_name == "time" and desired >= 11:
            values.append("00:00:00." + "0" * (desired - 10) + "Z")
        elif format_name == "duration" and desired >= 3:
            values.append("P" + "1" * (desired - 2) + "D")
        elif format_name == "ipv4" and 7 <= desired <= 15:
            extra_digits = desired - 7
            octet_lengths = [1, 1, 1, 1]
            for index in range(4):
                added = min(2, extra_digits)
                octet_lengths[index] += added
                extra_digits -= added
            values.append(".".join("1" * length for length in octet_lengths))
    if format_name == "ipv6":
        values.extend(("::1", "2001:db8::1", "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"))
    return tuple(values)


def _fraction(value: object) -> Fraction:
    if type(value) is int:
        return Fraction(value)
    if type(value) is float and math.isfinite(value):
        return Fraction(str(value))
    raise _WitnessGenerationError


def _numeric_bounds(
    schema: dict[str, object],
) -> tuple[tuple[Fraction, bool] | None, tuple[Fraction, bool] | None]:
    lower: tuple[Fraction, bool] | None = None
    upper: tuple[Fraction, bool] | None = None
    for keyword, exclusive in (("minimum", False), ("exclusiveMinimum", True)):
        if keyword not in schema:
            continue
        current = (_fraction(schema[keyword]), exclusive)
        if lower is None or current[0] > lower[0] or (
            current[0] == lower[0] and current[1]
        ):
            lower = current
    for keyword, exclusive in (("maximum", False), ("exclusiveMaximum", True)):
        if keyword not in schema:
            continue
        current = (_fraction(schema[keyword]), exclusive)
        if upper is None or current[0] < upper[0] or (
            current[0] == upper[0] and current[1]
        ):
            upper = current
    return lower, upper


def _lower_lattice_index(bound: tuple[Fraction, bool], step: Fraction) -> int:
    ratio = bound[0] / step
    index = math.ceil(ratio)
    return index + 1 if bound[1] and ratio.denominator == 1 else index


def _upper_lattice_index(bound: tuple[Fraction, bool], step: Fraction) -> int:
    ratio = bound[0] / step
    index = math.floor(ratio)
    return index - 1 if bound[1] and ratio.denominator == 1 else index


def _fraction_json_number(value: Fraction, *, integer: bool) -> int | float:
    if integer:
        if value.denominator != 1 or not -_MAX_SAFE_INTEGER <= value.numerator <= _MAX_SAFE_INTEGER:
            raise _WitnessGenerationError
        return value.numerator
    if value.denominator == 1 and -_MAX_SAFE_INTEGER <= value.numerator <= _MAX_SAFE_INTEGER:
        return value.numerator
    converted = float(value)
    if not math.isfinite(converted):
        raise _WitnessGenerationError
    return converted


def _lattice_numeric_candidates(
    schema: dict[str, object],
    *,
    integer: bool,
) -> tuple[int | float, ...]:
    lower, upper = _numeric_bounds(schema)
    declared_multiple = schema.get("multipleOf", 1)
    if integer:
        multiple = _fraction(declared_multiple)
        step = Fraction(abs(multiple.numerator))
    else:
        step = _fraction(declared_multiple)

    lower_index = _lower_lattice_index(lower, step) if lower is not None else None
    upper_index = _upper_lattice_index(upper, step) if upper is not None else None
    if lower_index is not None and lower_index > 0:
        closest = lower_index
    elif upper_index is not None and upper_index < 0:
        closest = upper_index
    else:
        closest = 0

    indices = {closest + offset for offset in range(-2, 3)}
    if lower_index is not None:
        indices.update((lower_index - 1, lower_index, lower_index + 1))
    if upper_index is not None:
        indices.update((upper_index - 1, upper_index, upper_index + 1))
    candidates: list[int | float] = []
    for index in sorted(indices):
        try:
            candidates.append(_fraction_json_number(step * index, integer=integer))
        except _WitnessGenerationError:
            continue
        if not integer:
            operational = cast(int | float, declared_multiple) * index
            if (
                (
                    type(operational) is int
                    and -_MAX_SAFE_INTEGER <= operational <= _MAX_SAFE_INTEGER
                )
                or (type(operational) is float and math.isfinite(operational))
            ):
                candidates.append(operational)
    return tuple(candidates[:_MAX_WITNESS_NUMERIC_CANDIDATES])


def _free_number_candidates(schema: dict[str, object]) -> tuple[int | float, ...]:
    lower, upper = _numeric_bounds(schema)
    values: list[int | float] = [0, 0.5, -0.5, 1, -1]
    for bound in (lower, upper):
        if bound is None:
            continue
        try:
            exact = _fraction_json_number(bound[0], integer=False)
            values.append(exact)
            values.extend(
                (
                    math.nextafter(float(exact), -math.inf),
                    math.nextafter(float(exact), math.inf),
                )
            )
        except (OverflowError, _WitnessGenerationError):
            continue
    return tuple(values[:_MAX_WITNESS_NUMERIC_CANDIDATES])


def _prioritized_witness_candidates(
    candidates: _WitnessCandidateSet,
) -> tuple[JSONValue, ...]:
    prioritized: list[JSONValue] = []
    seen: set[bytes] = set()
    aggregate_bytes = 0
    for group in (candidates.valid, candidates.raw):
        for candidate in group:
            canonical = canonical_json_bytes(candidate)
            if canonical in seen:
                continue
            aggregate_bytes += len(canonical)
            if aggregate_bytes > _MAX_WITNESS_CANDIDATE_POOL_BYTES:
                raise _WitnessGenerationError
            seen.add(canonical)
            prioritized.append(candidate)
            if len(prioritized) >= _MAX_WITNESS_CANDIDATES:
                return tuple(prioritized)
    return tuple(prioritized)


def _offer_bounded_witness_candidate(
    pool: dict[bytes, JSONValue],
    canonical: bytes,
    candidate: JSONValue,
) -> None:
    if canonical in pool:
        return
    pool[canonical] = candidate
    if len(pool) <= _MAX_WITNESS_CANDIDATES:
        return
    worst = max(
        pool,
        key=lambda key: (*_candidate_preference(pool[key]), key),
    )
    del pool[worst]


def _raw_witness_candidates_once(
    schema: object,
    format_entries: tuple[tuple[str, _FormatCheckerEntry], ...],
    depth: int,
) -> tuple[JSONValue, ...]:
    if depth > _MAX_WITNESS_SCHEMA_DEPTH:
        raise _WitnessGenerationError
    if type(schema) is bool:
        return (None,) if schema else ()
    if type(schema) is not dict:
        raise _WitnessGenerationError

    if "const" in schema:
        return _bounded_distinct_candidates((schema["const"],))
    if "enum" in schema:
        enum_values = schema["enum"]
        if type(enum_values) is not list:
            raise _WitnessGenerationError
        return _bounded_distinct_candidates(enum_values)
    if "anyOf" in schema or "oneOf" in schema:
        raise _WitnessGenerationError

    effective_type = _witness_effective_type(schema)
    if effective_type is None:
        return (None,)
    type_value = schema.get("type")
    nullable = type(type_value) is list
    candidates: list[object] = [None] if nullable else []

    if effective_type == "null":
        candidates.append(None)
    elif effective_type == "boolean":
        candidates.extend((False, True))
    elif effective_type == "string":
        minimum_length = cast(int, schema.get("minLength", 0))
        maximum_length = schema.get("maxLength")
        length_hints = {minimum_length}
        if minimum_length < _MAX_WITNESS_STRING_LENGTH:
            length_hints.add(minimum_length + 1)
        if type(maximum_length) is int:
            if maximum_length <= _MAX_WITNESS_STRING_LENGTH:
                length_hints.add(maximum_length)
            if maximum_length < _MAX_WITNESS_STRING_LENGTH:
                length_hints.add(maximum_length + 1)
        format_name = schema.get("format")
        if format_name is None:
            candidates.extend("a" * length for length in sorted(length_hints))
        elif type(format_name) is str:
            candidates.extend(
                _string_format_candidates(
                    format_name,
                    minimum_length,
                    cast(int | None, maximum_length),
                )
            )
        else:
            raise _WitnessGenerationError
    elif effective_type == "array":
        minimum_items = cast(int, schema.get("minItems", 0))
        maximum_items = schema.get("maxItems")
        item_schema = schema.get("items", True)
        item_candidate_set = _witness_candidate_set(
            item_schema,
            format_entries,
            depth + 1,
        )
        item_candidates = _prioritized_witness_candidates(item_candidate_set)
        lengths = {minimum_items}
        if minimum_items < _MAX_WITNESS_ARRAY_ITEMS:
            lengths.add(minimum_items + 1)
        if type(maximum_items) is int:
            if maximum_items <= _MAX_WITNESS_ARRAY_ITEMS:
                lengths.add(maximum_items)
            if maximum_items < _MAX_WITNESS_ARRAY_ITEMS:
                lengths.add(maximum_items + 1)
        ordered_lengths = sorted(lengths)
        items_per_length = max(
            1,
            (_MAX_WITNESS_CANDIDATES - len(candidates)) // len(ordered_lengths),
        )
        for length in ordered_lengths:
            if length == 0:
                _project_uniform_array_budget(None, 0)
                candidates.append(_materialize_uniform_array(None, 0))
                continue
            for item in item_candidates[:items_per_length]:
                if len(candidates) >= _MAX_WITNESS_CANDIDATES:
                    break
                _project_uniform_array_budget(item, length)
                candidates.append(_materialize_uniform_array(item, length))
            if len(candidates) >= _MAX_WITNESS_CANDIDATES:
                break
    elif effective_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if type(properties) is not dict or type(required) is not list:
            raise _WitnessGenerationError
        witness: dict[str, JSONValue] = {}
        required_candidates: dict[str, tuple[JSONValue, ...]] = {}
        for name in sorted(required):
            property_candidate_set = _witness_candidate_set(
                properties[name],
                format_entries,
                depth + 1,
            )
            if not property_candidate_set.valid:
                return ()
            required_candidates[name] = _prioritized_witness_candidates(
                property_candidate_set
            )
            selected = property_candidate_set.valid[0]
            _project_object_candidate_budget(witness, name, selected)
            witness = _materialize_object_candidate(witness, name, selected)
        candidates.append(witness)
        for name in sorted(required_candidates):
            for alternative in required_candidates[name][1:]:
                if len(candidates) >= _MAX_WITNESS_CANDIDATES:
                    break
                _project_object_candidate_budget(witness, name, alternative)
                neighbor = _materialize_object_candidate(
                    witness,
                    name,
                    alternative,
                )
                candidates.append(neighbor)
        for name in sorted(set(properties) - set(required)):
            if len(candidates) >= _MAX_WITNESS_CANDIDATES:
                break
            property_candidate_set = _witness_candidate_set(
                properties[name],
                format_entries,
                depth + 1,
            )
            property_candidates = _prioritized_witness_candidates(
                property_candidate_set
            )
            for property_candidate in property_candidates:
                if len(candidates) >= _MAX_WITNESS_CANDIDATES:
                    break
                _project_object_candidate_budget(witness, name, property_candidate)
                neighbor = _materialize_object_candidate(
                    witness,
                    name,
                    property_candidate,
                )
                candidates.append(neighbor)
    elif effective_type == "integer":
        candidates.extend(_lattice_numeric_candidates(schema, integer=True))
    elif effective_type == "number":
        if "multipleOf" in schema:
            candidates.extend(_lattice_numeric_candidates(schema, integer=False))
        else:
            candidates.extend(_free_number_candidates(schema))
    else:  # pragma: no cover - effective type is exhaustively audited.
        raise _WitnessGenerationError
    return _bounded_distinct_candidates(candidates)


def _witness_candidate_set(
    schema: object,
    format_entries: tuple[tuple[str, _FormatCheckerEntry], ...],
    depth: int = 0,
) -> _WitnessCandidateSet:
    try:
        validator = Draft202012Validator(
            schema,
            format_checker=_new_format_checker(format_entries),
            registry=Registry(),
        )
        if type(schema) is dict and ("anyOf" in schema or "oneOf" in schema):
            combinator = "anyOf" if "anyOf" in schema else "oneOf"
            branches = schema[combinator]
            if type(branches) is not list:
                raise _WitnessGenerationError
            raw_pool: dict[bytes, JSONValue] = {}
            valid_pool: dict[bytes, JSONValue] = {}
            aggregate_bytes = 0
            for branch in branches:
                branch_candidates = _witness_candidate_set(
                    branch,
                    format_entries,
                    depth + 1,
                )
                if combinator == "anyOf":
                    prioritized = branch_candidates.valid[:1]
                else:
                    prioritized = _prioritized_witness_candidates(
                        branch_candidates
                    )
                for candidate in prioritized:
                    canonical = canonical_json_bytes(candidate)
                    aggregate_bytes += len(canonical)
                    if aggregate_bytes > _MAX_WITNESS_CANDIDATE_POOL_BYTES:
                        raise _WitnessGenerationError
                    _offer_bounded_witness_candidate(
                        raw_pool,
                        canonical,
                        candidate,
                    )
                    if validator.is_valid(candidate):
                        _offer_bounded_witness_candidate(
                            valid_pool,
                            canonical,
                            candidate,
                        )
            raw = tuple(
                sorted(raw_pool.values(), key=_candidate_sort_key)[
                    :_MAX_WITNESS_CANDIDATES
                ]
            )
            valid = tuple(
                sorted(valid_pool.values(), key=_candidate_sort_key)[
                    :_MAX_WITNESS_CANDIDATES
                ]
            )
            return _WitnessCandidateSet(raw, valid)

        generated = _raw_witness_candidates_once(schema, format_entries, depth)
        valid = tuple(
            sorted(
                (
                    candidate
                    for candidate in generated
                    if validator.is_valid(candidate)
                ),
                key=_candidate_sort_key,
            )[:_MAX_WITNESS_CANDIDATES]
        )
        return _WitnessCandidateSet(
            generated[:_MAX_WITNESS_CANDIDATES],
            valid,
        )
    except _AdapterIntegrityError:
        raise
    except Exception:
        raise _WitnessGenerationError from None


class AppWorldSemanticAdapter(SemanticAdapter):
    """Immutable snapshot of one world's full non-admin AppWorld catalog."""

    __slots__ = (
        "_binding_runtime_seal_lookup",
        "_binding_runtime_seal_lookup_seal",
        "_binding_runtime_seals",
        "_binding_runtime_seals_seal",
        "_bindings",
        "_bindings_seal",
        "_catalog_canonical",
        "_catalog_canonical_seal",
        "_catalog_snapshot",
        "_catalog_snapshot_seal",
        "_integrity_token",
        "_validators",
        "_validators_seal",
        "_variant_id_seal",
        "_variant_manifest_canonical_seal",
        "_variant_manifest_root_seal",
        "_variant_seal",
        "_variant_tools_seal",
    )

    def __init__(
        self,
        function_catalog: Sequence[Mapping[str, object]],
    ) -> None:
        entries, catalog_snapshot, catalog_canonical = _snapshot_catalog(function_catalog)
        records = _catalog_records(entries)

        tools: list[SurfaceToolSpec] = []
        bindings: dict[str, _ToolBinding] = {}
        validators: dict[str, Draft202012Validator] = {}
        try:
            for name, app_name, api_name, description, schema, format_entries in records:
                tool = SurfaceToolSpec(name, description, schema)
                validator = Draft202012Validator(
                    tool.input_schema,
                    format_checker=_new_format_checker(format_entries),
                    registry=Registry(),
                )
                tools.append(tool)
                validators[name] = validator
                bindings[name] = _ToolBinding(
                    app_name,
                    api_name,
                    tool.input_schema,
                    canonical_json_bytes(tool.input_schema),
                    validator,
                )
            variant = SchemaVariant(_VARIANT_ID, tuple(tools), _VARIANT_MANIFEST)
        except Exception:
            raise ValueError(_CATALOG_ERROR) from None

        super().__init__(variant)
        validator_root = MappingProxyType(validators)
        binding_root = MappingProxyType(bindings)
        try:
            runtime_seals = tuple(
                _ToolRuntimeSeal(
                    position=position,
                    name=name,
                    app_name=binding.app_name,
                    api_name=binding.api_name,
                    tool=tool,
                    description=tool.description,
                    schema=binding.schema,
                    schema_canonical=binding.schema_canonical,
                    schema_root=_make_frozen_mapping_root_seal(binding.schema),
                    binding=binding,
                    validator_seal=_make_validator_seal(binding.validator),
                )
                for position, tool in enumerate(variant.tools)
                for name, binding in ((tool.name, bindings[tool.name]),)
            )
            runtime_seal_lookup = MappingProxyType(
                {seal.name: seal for seal in runtime_seals}
            )
            if len(runtime_seal_lookup) != len(runtime_seals):
                raise ValueError
            variant_manifest_root = _make_frozen_mapping_root_seal(variant.manifest)
        except Exception:
            raise ValueError(_CATALOG_ERROR) from None

        self._variant_seal = variant
        self._variant_id_seal = variant.variant_id
        self._variant_tools_seal = variant.tools
        self._variant_manifest_root_seal = variant_manifest_root
        self._variant_manifest_canonical_seal = object.__getattribute__(
            variant,
            "_manifest_canonical",
        )
        self._catalog_snapshot = catalog_snapshot
        self._catalog_snapshot_seal = catalog_snapshot
        self._catalog_canonical = catalog_canonical
        self._catalog_canonical_seal = catalog_canonical
        self._validators = validator_root
        self._validators_seal = validator_root
        self._bindings = binding_root
        self._bindings_seal = binding_root
        self._binding_runtime_seals = runtime_seals
        self._binding_runtime_seals_seal = runtime_seals
        self._binding_runtime_seal_lookup = runtime_seal_lookup
        self._binding_runtime_seal_lookup_seal = runtime_seal_lookup
        self._integrity_token = _APPWORLD_ADAPTER_INTEGRITY_TOKEN
        self._require_integrity()

    def _require_integrity(self) -> None:
        try:
            variant = object.__getattribute__(self, "_variant")
            variant_seal = object.__getattribute__(self, "_variant_seal")
            tools = object.__getattribute__(variant_seal, "tools")
            manifest = object.__getattribute__(variant_seal, "manifest")
            runtime_seals = object.__getattribute__(self, "_binding_runtime_seals")
            runtime_seal_lookup = object.__getattribute__(
                self,
                "_binding_runtime_seal_lookup",
            )
            bindings = object.__getattribute__(self, "_bindings")
            validators = object.__getattribute__(self, "_validators")
            # The detached catalog is construction provenance and is never read on the
            # rollout path, so its immutable tuple root and canonical-cache slot are
            # identity-attested here. The selected behavior-bearing schema is checked
            # deeply below; scanning every protected catalog schema per call would make
            # runtime cost grow with the full interface rather than the selected tool.
            if (
                object.__getattribute__(self, "_integrity_token")
                is not _APPWORLD_ADAPTER_INTEGRITY_TOKEN
                or type(variant) is not SchemaVariant
                or variant is not variant_seal
                or object.__getattribute__(variant_seal, "variant_id")
                != object.__getattribute__(self, "_variant_id_seal")
                or tools is not object.__getattribute__(self, "_variant_tools_seal")
                or type(tools) is not tuple
                or manifest
                is not object.__getattribute__(
                    self,
                    "_variant_manifest_root_seal",
                ).mapping
                or not _frozen_mapping_root_matches(
                    object.__getattribute__(self, "_variant_manifest_root_seal")
                )
                or object.__getattribute__(variant_seal, "_manifest_canonical")
                != object.__getattribute__(self, "_variant_manifest_canonical_seal")
                or canonical_json_bytes(manifest)
                != object.__getattribute__(self, "_variant_manifest_canonical_seal")
                or object.__getattribute__(self, "_catalog_snapshot")
                is not object.__getattribute__(self, "_catalog_snapshot_seal")
                or type(object.__getattribute__(self, "_catalog_snapshot")) is not tuple
                or object.__getattribute__(self, "_catalog_canonical")
                != object.__getattribute__(self, "_catalog_canonical_seal")
                or type(object.__getattribute__(self, "_catalog_canonical")) is not bytes
                or bindings is not object.__getattribute__(self, "_bindings_seal")
                or validators is not object.__getattribute__(self, "_validators_seal")
                or runtime_seals
                is not object.__getattribute__(self, "_binding_runtime_seals_seal")
                or type(runtime_seals) is not tuple
                or runtime_seal_lookup
                is not object.__getattribute__(
                    self,
                    "_binding_runtime_seal_lookup_seal",
                )
                or type(runtime_seal_lookup) is not MappingProxyType
                or len(tools) != len(runtime_seals)
                or len(runtime_seal_lookup) != len(runtime_seals)
                or len(bindings) != len(runtime_seals)
                or len(validators) != len(runtime_seals)
            ):
                raise _AdapterIntegrityError(_INTEGRITY_ERROR)
        except _AdapterIntegrityError:
            raise
        except Exception:
            raise _AdapterIntegrityError(_INTEGRITY_ERROR) from None

    def _runtime_seal_for_name(self, name: str) -> _ToolRuntimeSeal:
        try:
            lookup = object.__getattribute__(self, "_binding_runtime_seal_lookup")
            seal = lookup.get(name)
            if seal is None:
                raise ValueError(_SURFACE_CALL_ERROR)
            if type(seal) is not _ToolRuntimeSeal or seal.name != name:
                raise _AdapterIntegrityError(_INTEGRITY_ERROR)
            self._require_selected_schema_integrity(seal)
            return seal
        except _AdapterIntegrityError:
            raise
        except Exception:
            raise ValueError(_SURFACE_CALL_ERROR) from None

    def _require_selected_schema_integrity(self, seal: _ToolRuntimeSeal) -> None:
        try:
            tools = object.__getattribute__(self, "_variant_tools_seal")
            bindings = object.__getattribute__(self, "_bindings")
            validators = object.__getattribute__(self, "_validators")
            validator = seal.validator_seal.validator
            if (
                type(seal.position) is not int
                or seal.position < 0
                or seal.position >= len(tools)
                or tools[seal.position] is not seal.tool
                or type(seal.tool) is not SurfaceToolSpec
                or object.__getattribute__(seal.tool, "name") != seal.name
                or object.__getattribute__(seal.tool, "description") != seal.description
                or object.__getattribute__(seal.tool, "input_schema") is not seal.schema
                or object.__getattribute__(seal.tool, "_input_schema_canonical")
                != seal.schema_canonical
                or not _frozen_mapping_root_matches(seal.schema_root)
                or bindings.get(seal.name) is not seal.binding
                or validators.get(seal.name) is not validator
                or type(seal.binding) is not _ToolBinding
                or object.__getattribute__(seal.binding, "app_name") != seal.app_name
                or object.__getattribute__(seal.binding, "api_name") != seal.api_name
                or f"{seal.app_name}__{seal.api_name}" != seal.name
                or object.__getattribute__(seal.binding, "schema") is not seal.schema
                or object.__getattribute__(seal.binding, "schema_canonical")
                != seal.schema_canonical
                or object.__getattribute__(seal.binding, "validator") is not validator
                or object.__getattribute__(validator, "schema") is not seal.schema
                or canonical_json_bytes(seal.schema) != seal.schema_canonical
                or not _validator_seal_matches(seal.validator_seal)
            ):
                raise _AdapterIntegrityError(_INTEGRITY_ERROR)
        except _AdapterIntegrityError:
            raise
        except Exception:
            raise _AdapterIntegrityError(_INTEGRITY_ERROR) from None

    def _guarded(
        self,
        operation: Callable[[], _ResultT],
        error_message: str,
    ) -> _ResultT:
        self._require_integrity()
        try:
            try:
                return operation()
            except _AdapterIntegrityError:
                raise ValueError(_INTEGRITY_ERROR) from None
            except Exception:
                raise ValueError(error_message) from None
        finally:
            self._require_integrity()

    def _validate_arguments(
        self,
        seal: _ToolRuntimeSeal,
        arguments: Mapping[str, JSONValue],
        error_message: str,
    ) -> None:
        self._require_selected_schema_integrity(seal)
        try:
            plain_arguments = _plain_json(
                cast(JSONValue, arguments),
                error_message,
            )
            if type(plain_arguments) is not dict:
                raise ValueError(error_message)
            seal.validator_seal.validator.validate(plain_arguments)
        finally:
            self._require_selected_schema_integrity(seal)

    def _parse_surface_call(
        self,
        surface_call: object,
        error_message: str,
    ) -> SemanticAction:
        call = _snapshot_runtime_mapping(surface_call, error_message)
        name = call.get("name")
        arguments = call.get("arguments")
        if type(name) is not str:
            raise ValueError(error_message)
        seal = self._runtime_seal_for_name(name)
        if set(call) != _CALL_KEYS or not isinstance(arguments, Mapping):
            raise ValueError(error_message)
        self._validate_arguments(
            seal,
            cast(Mapping[str, JSONValue], arguments),
            error_message,
        )
        return SemanticAction(name, cast(Mapping[str, JSONValue], arguments))

    def _validate_action(
        self,
        action: object,
        error_message: str,
    ) -> SemanticAction:
        rebuilt = _action_snapshot(action, error_message)
        seal = self._runtime_seal_for_name(rebuilt.name)
        self._validate_arguments(seal, rebuilt.arguments, error_message)
        return rebuilt

    def _compile_action(
        self,
        action: object,
        error_message: str,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        rebuilt = self._validate_action(action, error_message)
        call = _snapshot_runtime_mapping(
            {"name": rebuilt.name, "arguments": rebuilt.arguments},
            error_message,
        )
        return (call,)

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        return self._guarded(
            lambda: (self._parse_surface_call(surface_call, _SURFACE_CALL_ERROR),),
            _SURFACE_CALL_ERROR,
        )

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        return self._guarded(
            lambda: self._compile_action(action, _SEMANTIC_ACTION_ERROR),
            _SEMANTIC_ACTION_ERROR,
        )

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        def wrap() -> JSONValue:
            parsed = self._parse_surface_call(surface_call, _OBSERVATION_ERROR)
            if type(actions) is not tuple or len(actions) != 1:
                raise ValueError(_OBSERVATION_ERROR)
            recorded = self._validate_action(actions[0], _OBSERVATION_ERROR)
            if not _actions_equal(parsed, recorded):
                raise ValueError(_OBSERVATION_ERROR)
            if (
                type(base_observation_groups) is not tuple
                or len(base_observation_groups) != 1
                or type(base_observation_groups[0]) is not tuple
                or len(base_observation_groups[0]) != 1
            ):
                raise ValueError(_OBSERVATION_ERROR)
            seal = self._runtime_seal_for_name(parsed.name)
            try:
                return _snapshot_runtime_json(
                    base_observation_groups[0][0],
                    _OBSERVATION_ERROR,
                )
            finally:
                self._require_selected_schema_integrity(seal)

        return self._guarded(wrap, _OBSERVATION_ERROR)

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        def canonicalize() -> tuple[SemanticAction, ...]:
            if not _execution_trace_has_canonical_shape(trace):
                raise ValueError(_TRACE_ERROR)
            typed_trace = cast(ExecutionTrace, trace)
            surface_calls = object.__getattribute__(typed_trace, "surface_calls")
            semantic_actions = object.__getattribute__(typed_trace, "semantic_actions")
            base_calls = object.__getattribute__(typed_trace, "base_calls")
            if (
                canonical_json_bytes(cast(JSONValue, surface_calls))
                != object.__getattribute__(typed_trace, "_surface_calls_canonical")
                or canonical_json_bytes(cast(JSONValue, base_calls))
                != object.__getattribute__(typed_trace, "_base_calls_canonical")
            ):
                raise ValueError(_TRACE_ERROR)

            parsed_actions = tuple(
                self._parse_surface_call(call, _TRACE_ERROR) for call in surface_calls
            )
            if len(semantic_actions) != len(parsed_actions):
                raise ValueError(_TRACE_ERROR)
            rebuilt_recorded = tuple(
                self._validate_action(action, _TRACE_ERROR)
                for action in semantic_actions
            )
            if any(
                not _actions_equal(parsed, recorded)
                for parsed, recorded in zip(
                    parsed_actions,
                    rebuilt_recorded,
                    strict=True,
                )
            ):
                raise ValueError(_TRACE_ERROR)

            expected_base_calls = tuple(
                self._compile_action(action, _TRACE_ERROR)[0]
                for action in parsed_actions
            )
            if len(base_calls) != len(expected_base_calls):
                raise ValueError(_TRACE_ERROR)
            if any(
                canonical_json_bytes(cast(JSONValue, actual))
                != canonical_json_bytes(cast(JSONValue, expected))
                for actual, expected in zip(
                    base_calls,
                    expected_base_calls,
                    strict=True,
                )
            ):
                raise ValueError(_TRACE_ERROR)
            return parsed_actions

        return self._guarded(canonicalize, _TRACE_ERROR)


def build_appworld_adapter(
    function_catalog: Sequence[Mapping[str, object]],
) -> AppWorldSemanticAdapter:
    """Build one pure adapter from a detached AppWorld function catalog snapshot."""

    return AppWorldSemanticAdapter(function_catalog)


def build_minimal_source_calls(
    source_adapter: AppWorldSemanticAdapter,
) -> tuple[Mapping[str, JSONValue], ...]:
    """Build one deterministic, side-effect-free schema witness per source tool."""

    if type(source_adapter) is not AppWorldSemanticAdapter:
        raise ValueError(_WITNESS_ERROR)

    final_integrity_seals: list[_ToolRuntimeSeal] = []

    def build() -> tuple[Mapping[str, JSONValue], ...]:
        try:
            runtime_seals = object.__getattribute__(
                source_adapter,
                "_binding_runtime_seals",
            )
            if type(runtime_seals) is not tuple or not runtime_seals:
                raise _WitnessGenerationError
            final_integrity_seals.extend(runtime_seals)

            audited: list[tuple[_ToolRuntimeSeal, dict[str, object]]] = []
            for seal in runtime_seals:
                source_adapter._require_selected_schema_integrity(seal)
                try:
                    schema = _plain_json(cast(JSONValue, seal.schema), _WITNESS_ERROR)
                    if type(schema) is not dict:
                        raise _WitnessGenerationError
                    _audit_witness_schema(schema)
                    audited.append((seal, schema))
                finally:
                    source_adapter._require_selected_schema_integrity(seal)

            calls: list[Mapping[str, JSONValue]] = []
            for seal, schema in audited:
                source_adapter._require_selected_schema_integrity(seal)
                try:
                    format_entries = _schema_format_entries(schema)
                    candidates = _witness_candidate_set(
                        schema,
                        format_entries,
                    ).valid
                    if not candidates or type(candidates[0]) is not dict:
                        raise _WitnessGenerationError
                    arguments = cast(dict[str, JSONValue], candidates[0])

                    fresh_validator = Draft202012Validator(
                        schema,
                        format_checker=_new_format_checker(format_entries),
                        registry=Registry(),
                    )
                    fresh_arguments = _plain_witness_value(arguments)
                    if type(fresh_arguments) is not dict:
                        raise _WitnessGenerationError
                    fresh_validator.validate(fresh_arguments)

                    frozen_arguments = _snapshot_runtime_mapping(
                        arguments,
                        _WITNESS_ERROR,
                    )
                    source_adapter._validate_arguments(
                        seal,
                        frozen_arguments,
                        _WITNESS_ERROR,
                    )
                    call = _snapshot_runtime_mapping(
                        {"name": seal.name, "arguments": frozen_arguments},
                        _WITNESS_ERROR,
                    )
                    if set(call) != _CALL_KEYS:
                        raise _WitnessGenerationError
                    calls.append(call)
                finally:
                    source_adapter._require_selected_schema_integrity(seal)

            for (seal, _schema), call in zip(audited, calls, strict=True):
                source_adapter._require_selected_schema_integrity(seal)
                try:
                    actions = source_adapter.surface_to_semantic(call)
                    if type(actions) is not tuple or len(actions) != 1:
                        raise _WitnessGenerationError
                    expected_arguments = call.get("arguments")
                    if not isinstance(expected_arguments, Mapping) or not _actions_equal(
                        actions[0],
                        SemanticAction(seal.name, expected_arguments),
                    ):
                        raise _WitnessGenerationError
                finally:
                    source_adapter._require_selected_schema_integrity(seal)
            return tuple(calls)
        except (_AdapterIntegrityError, _WitnessGenerationError):
            raise
        except Exception:
            raise _WitnessGenerationError from None

    def build_with_full_integrity_check() -> tuple[Mapping[str, JSONValue], ...]:
        try:
            return build()
        finally:
            for seal in final_integrity_seals:
                source_adapter._require_selected_schema_integrity(seal)

    return source_adapter._guarded(build_with_full_integrity_check, _WITNESS_ERROR)


__all__ = [
    "AppWorldSemanticAdapter",
    "build_appworld_adapter",
    "build_minimal_source_calls",
]
