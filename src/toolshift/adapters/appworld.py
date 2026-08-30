"""Pure catalog snapshots for the external AppWorld runtime."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar, cast

from jsonschema import Draft202012Validator, FormatChecker

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


def _declared_format_entries(
    schema: dict[str, object],
) -> tuple[tuple[str, _FormatCheckerEntry], ...]:
    declared: set[str] = set()

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
            if "format" in node:
                format_name = node["format"]
                if (
                    type(format_name) is not str
                    or format_name not in _SUPPORTED_DRAFT202012_FORMATS
                ):
                    raise _SchemaSnapshotError
                declared.add(format_name)

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
        for format_name in sorted(declared):
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
        return schema, _declared_format_entries(schema)
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
                raise
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


__all__ = ["AppWorldSemanticAdapter", "build_appworld_adapter"]
