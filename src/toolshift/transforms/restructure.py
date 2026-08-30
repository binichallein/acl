"""Deterministic L2 grouping of selected top-level tool parameters."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.schema import schema_fingerprint
from toolshift.transforms._runtime import (
    _delegate_with_binding_guard,
    _make_mapping_root_seal,
    _make_operator_runtime_seal,
    _make_schema_runtime_seal,
    _mapping_root_seal_matches,
    _MappingRootSeal,
    _operator_runtime_seal_matches,
    _OperatorRuntimeSeal,
    _raw_adapter_variant,
    _schema_runtime_seal_matches,
    _SchemaRuntimeSeal,
)
from toolshift.transforms.base import (
    OperatorManifestEntry,
    TransformValidationError,
    build_transformed_variant,
    derive_operator_seed,
)
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    _execution_trace_has_canonical_shape,
    _freeze_mapping,
    canonical_json_bytes,
)

_PARAMETER_RESTRUCTURE_OPERATOR = "parameter_restructure"
_PARAMETER_RESTRUCTURE_LEVEL = "L2"
_PARAMETER_RESTRUCTURE_OPERATOR_ID = "restructure-000"
_PARAMETER_RESTRUCTURE_MODE = "nest_top_level"
_PARAMETER_RESTRUCTURE_ABI_TAG = b"toolshift.transform.parameter-restructure.v1"
_DRAFT_2020_12_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_COMPOSITION_UNSUPPORTED_MESSAGE = (
    "parameter restructure composition requires the explicit compose transform"
)
PARAMETER_RESTRUCTURE_VERSION_HASH = hashlib.sha256(_PARAMETER_RESTRUCTURE_ABI_TAG).hexdigest()
_PARAMETER_RESTRUCTURE_INTEGRITY_TOKEN = object()
_ResultT = TypeVar("_ResultT")

_ROOT_STRUCTURAL_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
    }
)
_ROOT_ANNOTATION_KEYS = ("$schema", "title", "description", "$comment")
_ALLOWED_ROOT_KEYS = _ROOT_STRUCTURAL_KEYS | frozenset(_ROOT_ANNOTATION_KEYS)
_FORBIDDEN_SCHEMA_KEYWORDS = frozenset(
    {
        "$ref",
        "$dynamicRef",
        "$recursiveRef",
        "$recursiveAnchor",
        "$id",
        "$anchor",
        "$dynamicAnchor",
        "default",
        "examples",
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
_ARRAY_SUBSCHEMA_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_MAPPING_SUBSCHEMA_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)


@dataclass(frozen=True, slots=True)
class _ToolTranslationPlan:
    tool_name: str
    container_name: str
    declared: frozenset[str]
    moved: frozenset[str]
    unmoved: frozenset[str]
    required: frozenset[str]
    outer_required: frozenset[str]
    inner_required: frozenset[str]
    required_count: int
    outer_required_count: int
    inner_required_count: int
    _runtime_snapshot: tuple[object, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_runtime_snapshot",
            (
                self.tool_name,
                self.container_name,
                self.declared,
                self.moved,
                self.unmoved,
                self.required,
                self.outer_required,
                self.inner_required,
                self.required_count,
                self.outer_required_count,
                self.inner_required_count,
            ),
        )

    def require_integrity(self) -> None:
        try:
            snapshot = object.__getattribute__(self, "_runtime_snapshot")
            if (
                type(snapshot) is not tuple
                or len(snapshot) != 11
                or object.__getattribute__(self, "tool_name") != snapshot[0]
                or object.__getattribute__(self, "container_name") != snapshot[1]
                or object.__getattribute__(self, "declared") is not snapshot[2]
                or object.__getattribute__(self, "moved") is not snapshot[3]
                or object.__getattribute__(self, "unmoved") is not snapshot[4]
                or object.__getattribute__(self, "required") is not snapshot[5]
                or object.__getattribute__(self, "outer_required") is not snapshot[6]
                or object.__getattribute__(self, "inner_required") is not snapshot[7]
                or object.__getattribute__(self, "required_count") != snapshot[8]
                or object.__getattribute__(self, "outer_required_count") != snapshot[9]
                or object.__getattribute__(self, "inner_required_count") != snapshot[10]
            ):
                raise TransformValidationError(
                    "parameter restructure transform integrity validation failed"
                )
        except Exception:
            raise TransformValidationError(
                "parameter restructure transform integrity validation failed"
            ) from None


class _TranslationPlanLookup(Mapping[str, _ToolTranslationPlan]):
    __slots__ = ("_index", "_items")

    def __init__(self, plans: tuple[_ToolTranslationPlan, ...]) -> None:
        if type(plans) is not tuple:
            raise TransformValidationError("parameter restructure translation plans are invalid")
        items = tuple((plan.tool_name, plan) for plan in plans)
        index = {name: position for position, (name, _) in enumerate(items)}
        if len(index) != len(items):
            raise TransformValidationError("parameter restructure translation plans are invalid")
        object.__setattr__(self, "_items", items)
        object.__setattr__(self, "_index", index)

    def __getattribute__(self, name: str) -> object:
        if name == "_index":
            raise AttributeError("translation plan lookup index is private")
        return object.__getattribute__(self, name)

    def __getitem__(self, key: str) -> _ToolTranslationPlan:
        items = object.__getattribute__(self, "_items")
        index = object.__getattribute__(self, "_index")
        if type(items) is not tuple or type(index) is not dict:
            raise KeyError(key)
        position = dict.get(index, key)
        if type(position) is not int or position < 0 or position >= len(items):
            raise KeyError(key)
        entry = items[position]
        if type(entry) is not tuple or len(entry) != 2 or entry[0] != key:
            raise KeyError(key)
        plan = entry[1]
        if type(plan) is not _ToolTranslationPlan:
            raise KeyError(key)
        return plan

    def __iter__(self) -> Iterator[str]:
        items = object.__getattribute__(self, "_items")
        if type(items) is not tuple:
            return iter(())
        return (entry[0] for entry in items)

    def __len__(self) -> int:
        items = object.__getattribute__(self, "_items")
        if type(items) is not tuple:
            return 0
        return len(items)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("translation plan lookup is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("translation plan lookup is immutable")


def _is_schema_node(value: object) -> bool:
    return type(value) is bool or isinstance(value, Mapping)


def _require_non_blank_utf8(value: object, message: str) -> str:
    if type(value) is not str or not value.strip():
        raise TransformValidationError(message)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise TransformValidationError(message) from None
    return value


def _rule_snapshot(
    tool_name: str,
    container_name: str,
    moved_parameters: tuple[str, ...],
) -> bytes:
    return canonical_json_bytes(
        {
            "tool_name": tool_name,
            "container_name": container_name,
            "moved_parameters": moved_parameters,
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class ParameterGroupRule:
    """One normalized, immutable top-level parameter grouping rule."""

    tool_name: str
    container_name: str
    moved_parameters: tuple[str, ...]
    _canonical_snapshot: bytes = field(init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        tool_name = _require_non_blank_utf8(
            self.tool_name,
            "tool_name must be a non-blank valid UTF-8 string",
        )
        container_name = _require_non_blank_utf8(
            self.container_name,
            "container_name must be a non-blank valid UTF-8 string",
        )
        supplied = self.moved_parameters
        if type(supplied) is not tuple or not supplied:
            raise TransformValidationError("moved_parameters must be an exact non-empty tuple")
        normalized: list[str] = []
        seen: set[str] = set()
        for value in supplied:
            name = _require_non_blank_utf8(
                value,
                "moved_parameters must contain non-blank valid UTF-8 strings",
            )
            if name in seen:
                raise TransformValidationError("moved_parameters must not contain duplicates")
            seen.add(name)
            normalized.append(name)
        moved_parameters = tuple(sorted(normalized))
        snapshot = _rule_snapshot(tool_name, container_name, moved_parameters)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "container_name", container_name)
        object.__setattr__(self, "moved_parameters", moved_parameters)
        object.__setattr__(self, "_canonical_snapshot", snapshot)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        typed = cast(ParameterGroupRule, other)
        return (
            self.tool_name == typed.tool_name
            and self.container_name == typed.container_name
            and self.moved_parameters == typed.moved_parameters
            and self._canonical_snapshot == typed._canonical_snapshot
        )


def _rebuild_rule(value: object) -> ParameterGroupRule:
    if type(value) is not ParameterGroupRule:
        raise TransformValidationError(
            "rules must be an exact non-empty tuple of ParameterGroupRule values"
        )
    rule = cast(ParameterGroupRule, value)
    try:
        rebuilt = ParameterGroupRule(
            object.__getattribute__(rule, "tool_name"),
            object.__getattribute__(rule, "container_name"),
            object.__getattribute__(rule, "moved_parameters"),
        )
        snapshot = object.__getattribute__(rule, "_canonical_snapshot")
        if type(snapshot) is not bytes or snapshot != rebuilt._canonical_snapshot:
            raise TransformValidationError("parameter group rule integrity validation failed")
        return rebuilt
    except Exception:
        raise TransformValidationError("parameter group rule integrity validation failed") from None


def _normalize_rules(value: object) -> tuple[ParameterGroupRule, ...]:
    if type(value) is not tuple or not value:
        raise TransformValidationError(
            "rules must be an exact non-empty tuple of ParameterGroupRule values"
        )
    rebuilt = tuple(_rebuild_rule(rule) for rule in value)
    tool_names = tuple(rule.tool_name for rule in rebuilt)
    if len(tool_names) != len(set(tool_names)):
        raise TransformValidationError("parameter restructure rules contain a duplicate tool")
    return tuple(sorted(rebuilt, key=lambda rule: rule.tool_name))


def _single_subschemas_contain_forbidden_keyword(value: Mapping[object, object]) -> bool:
    for key in _SINGLE_SUBSCHEMA_KEYS:
        if key not in value:
            continue
        child = value[key]
        if not _is_schema_node(child):
            return True
        if _schema_node_contains_forbidden_keyword(child):
            return True
    return False


def _array_subschemas_contain_forbidden_keyword(value: Mapping[object, object]) -> bool:
    for key in _ARRAY_SUBSCHEMA_KEYS:
        if key not in value:
            continue
        children = value[key]
        if type(children) not in (list, tuple):
            return True
        if any(
            not _is_schema_node(child) or _schema_node_contains_forbidden_keyword(child)
            for child in children
        ):
            return True
    return False


def _mapping_subschemas_contain_forbidden_keyword(value: Mapping[object, object]) -> bool:
    for key in _MAPPING_SUBSCHEMA_KEYS:
        if key not in value:
            continue
        children = value[key]
        if not isinstance(children, Mapping):
            return True
        if any(
            not _is_schema_node(child) or _schema_node_contains_forbidden_keyword(child)
            for child in children.values()
        ):
            return True
    return False


def _dependencies_contain_forbidden_keyword(value: Mapping[object, object]) -> bool:
    dependencies = value.get("dependencies")
    return isinstance(dependencies, Mapping) and any(
        _schema_node_contains_forbidden_keyword(child)
        for child in dependencies.values()
        if type(child) is bool or isinstance(child, Mapping)
    )


def _schema_node_uses_unsupported_dialect(value: Mapping[object, object]) -> bool:
    return "$schema" in value and value["$schema"] != _DRAFT_2020_12_SCHEMA_URI


def _schema_node_contains_forbidden_keyword(value: object) -> bool:
    if type(value) is bool:
        return False
    if not isinstance(value, Mapping):
        return False
    return (
        _schema_node_uses_unsupported_dialect(value)
        or any(key in _FORBIDDEN_SCHEMA_KEYWORDS for key in value)
        or _single_subschemas_contain_forbidden_keyword(value)
        or _array_subschemas_contain_forbidden_keyword(value)
        or _mapping_subschemas_contain_forbidden_keyword(value)
        or _dependencies_contain_forbidden_keyword(value)
    )


def _valid_property_name(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _valid_root_annotations(schema: Mapping[str, JSONValue]) -> bool:
    if _schema_node_uses_unsupported_dialect(schema):
        return False
    for key in _ROOT_ANNOTATION_KEYS:
        if key not in schema:
            continue
        value = schema[key]
        if type(value) is not str:
            return False
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return False
    return True


def _admit_changed_schema(
    schema: Mapping[str, JSONValue],
) -> tuple[Mapping[str, JSONValue], tuple[str, ...]]:
    try:
        if (
            set(schema) - _ALLOWED_ROOT_KEYS
            or schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
            or not _valid_root_annotations(schema)
        ):
            raise TransformValidationError("parameter restructure source schema is unsupported")
        properties = schema.get("properties")
        if not isinstance(properties, Mapping) or not properties:
            raise TransformValidationError("parameter restructure source schema is unsupported")
        for name, subschema in properties.items():
            if (
                not _valid_property_name(name)
                or not (type(subschema) is bool or isinstance(subschema, Mapping))
                or _schema_node_contains_forbidden_keyword(subschema)
            ):
                raise TransformValidationError("parameter restructure source schema is unsupported")
        raw_required = schema.get("required", ())
        if type(raw_required) not in (list, tuple):
            raise TransformValidationError("parameter restructure source schema is unsupported")
        required = tuple(raw_required)
        if (
            any(type(name) is not str for name in required)
            or len(required) != len(set(required))
            or any(name not in properties for name in required)
        ):
            raise TransformValidationError("parameter restructure source schema is unsupported")
        return cast(Mapping[str, JSONValue], properties), cast(tuple[str, ...], required)
    except TransformValidationError:
        raise
    except Exception:
        raise TransformValidationError(
            "parameter restructure source schema is unsupported"
        ) from None


def _rewrite_tool(tool: SurfaceToolSpec, rule: ParameterGroupRule) -> SurfaceToolSpec:
    properties, required = _admit_changed_schema(tool.input_schema)
    moved = set(rule.moved_parameters)
    if rule.container_name in properties:
        raise TransformValidationError(
            "parameter restructure container conflicts with source schema"
        )
    if any(name not in properties for name in rule.moved_parameters):
        raise TransformValidationError("parameter restructure rules do not match source schema")

    inner: dict[str, JSONValue] = {
        "type": "object",
        "properties": {name: properties[name] for name in rule.moved_parameters},
        "additionalProperties": False,
    }
    moved_required = tuple(name for name in required if name in moved)
    if moved_required:
        inner["required"] = moved_required

    outer_properties: dict[str, JSONValue] = {
        name: subschema for name, subschema in properties.items() if name not in moved
    }
    outer_properties[rule.container_name] = inner
    outer: dict[str, JSONValue] = {
        key: tool.input_schema[key] for key in _ROOT_ANNOTATION_KEYS if key in tool.input_schema
    }
    outer.update(
        {
            "type": "object",
            "properties": outer_properties,
            "required": (*(name for name in required if name not in moved), rule.container_name),
            "additionalProperties": False,
        }
    )
    try:
        return SurfaceToolSpec(tool.name, tool.description, outer)
    except Exception:
        raise TransformValidationError("parameter restructure schema rewrite failed") from None


def _build_restructured_tools(
    source_variant: SchemaVariant,
    rules: tuple[ParameterGroupRule, ...],
) -> tuple[SurfaceToolSpec, ...]:
    tools_by_name = {tool.name: tool for tool in source_variant.tools}
    if any(rule.tool_name not in tools_by_name for rule in rules):
        raise TransformValidationError("parameter restructure rule references an unknown tool")
    rules_by_name = {rule.tool_name: rule for rule in rules}
    return tuple(
        _rewrite_tool(tool, rules_by_name[tool.name]) if tool.name in rules_by_name else tool
        for tool in source_variant.tools
    )


def _build_translation_plans(
    source_variant: SchemaVariant,
    rules: tuple[ParameterGroupRule, ...],
) -> tuple[frozenset[str], frozenset[str], _TranslationPlanLookup]:
    tools_by_name = {tool.name: tool for tool in source_variant.tools}
    plans: list[_ToolTranslationPlan] = []
    for rule in rules:
        tool = tools_by_name[rule.tool_name]
        properties, required_names = _admit_changed_schema(tool.input_schema)
        declared = frozenset(properties)
        moved = frozenset(rule.moved_parameters)
        required = frozenset(required_names)
        unmoved = declared - moved
        outer_required = (required - moved) | frozenset((rule.container_name,))
        inner_required = required & moved
        plans.append(
            _ToolTranslationPlan(
                tool_name=rule.tool_name,
                container_name=rule.container_name,
                declared=declared,
                moved=moved,
                unmoved=unmoved,
                required=required,
                outer_required=outer_required,
                inner_required=inner_required,
                required_count=len(required),
                outer_required_count=len(outer_required),
                inner_required_count=len(inner_required),
            )
        )
    return (
        frozenset(tools_by_name),
        frozenset(rule.tool_name for rule in rules),
        _TranslationPlanLookup(tuple(plans)),
    )


def _freeze_parameters(value: object) -> Mapping[str, JSONValue]:
    try:
        return _freeze_mapping(value, "parameter_restructure_parameters")
    except Exception:
        raise TransformValidationError(
            "parameter restructure operator parameters are invalid"
        ) from None


def _operator_parameters(
    rules: tuple[ParameterGroupRule, ...],
) -> Mapping[str, JSONValue]:
    return _freeze_parameters(
        {
            "mode": _PARAMETER_RESTRUCTURE_MODE,
            "rules": tuple(
                {
                    "tool_name": rule.tool_name,
                    "container_name": rule.container_name,
                    "moves": tuple(
                        {
                            "source_path": (name,),
                            "surface_path": (rule.container_name, name),
                        }
                        for name in rule.moved_parameters
                    ),
                }
                for rule in rules
            ),
        }
    )


def _require_operator_matches_rules(
    operator: OperatorManifestEntry,
    operator_manifest: Mapping[str, JSONValue],
    rules: tuple[ParameterGroupRule, ...],
) -> None:
    if (
        operator.operator_id != _PARAMETER_RESTRUCTURE_OPERATOR_ID
        or operator.operator != _PARAMETER_RESTRUCTURE_OPERATOR
        or operator.level != _PARAMETER_RESTRUCTURE_LEVEL
        or operator.version_hash != PARAMETER_RESTRUCTURE_VERSION_HASH
    ):
        raise TransformValidationError("parameter restructure operator metadata is invalid")
    actual = operator_manifest.get("parameters")
    expected = _operator_parameters(rules)
    if not isinstance(actual, Mapping) or canonical_json_bytes(actual) != canonical_json_bytes(
        expected
    ):
        raise TransformValidationError(
            "parameter restructure operator parameters do not match rules"
        )


def _require_variant_matches_rules(
    source_variant: SchemaVariant,
    variant: SchemaVariant,
    operator: OperatorManifestEntry,
    rules: tuple[ParameterGroupRule, ...],
) -> None:
    try:
        seed = variant.manifest["seed"]
        if type(seed) is not int:
            raise TransformValidationError("parameter restructure manifest seed is invalid")
        expected_tools = _build_restructured_tools(source_variant, rules)
        expected_variant = build_transformed_variant(
            source_variant,
            expected_tools,
            seed=seed,
            operators=(operator,),
        )
        identity_names = {tool.name for tool in source_variant.tools} - {
            rule.tool_name for rule in rules
        }
        actual_tools = {tool.name: tool for tool in variant.tools}
        if any(
            actual_tools.get(name) is not tool
            for name, tool in (
                (source_tool.name, source_tool)
                for source_tool in source_variant.tools
                if source_tool.name in identity_names
            )
        ):
            raise TransformValidationError(
                "parameter restructure variant does not match operator manifest"
            )
        if variant != expected_variant:
            raise TransformValidationError(
                "parameter restructure variant does not match operator manifest"
            )
    except TransformValidationError:
        raise
    except Exception:
        raise TransformValidationError(
            "parameter restructure variant integrity validation failed"
        ) from None


@dataclass(frozen=True, slots=True)
class _ValidatedConstruction:
    rules: tuple[ParameterGroupRule, ...]
    source_fingerprint: str
    variant_fingerprint: str
    operator_manifest: Mapping[str, JSONValue]


def _validate_construction(
    source_variant: object,
    variant: object,
    operator: object,
    rules: object,
) -> _ValidatedConstruction:
    if type(source_variant) is not SchemaVariant or type(variant) is not SchemaVariant:
        raise TransformValidationError(
            "parameter restructure variants must be exact SchemaVariant values"
        )
    if type(operator) is not OperatorManifestEntry:
        raise TransformValidationError(
            "parameter restructure operator must be an OperatorManifestEntry"
        )
    normalized = _normalize_rules(rules)
    typed_source = cast(SchemaVariant, source_variant)
    typed_variant = cast(SchemaVariant, variant)
    typed_operator = cast(OperatorManifestEntry, operator)
    try:
        operator_manifest = typed_operator.as_manifest()
    except Exception:
        raise TransformValidationError(
            "parameter restructure operator integrity validation failed"
        ) from None
    try:
        source_fingerprint = schema_fingerprint(typed_source)
        variant_fingerprint = schema_fingerprint(typed_variant)
    except Exception:
        raise TransformValidationError(
            "parameter restructure variant integrity validation failed"
        ) from None
    if typed_source.manifest.get("kind") == "toolshift_interface_variant":
        raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
    _require_operator_matches_rules(typed_operator, operator_manifest, normalized)
    _require_variant_matches_rules(typed_source, typed_variant, typed_operator, normalized)
    return _ValidatedConstruction(
        normalized,
        source_fingerprint,
        variant_fingerprint,
        operator_manifest,
    )


def _transform_snapshot(
    source_fingerprint: str,
    variant_fingerprint: str,
    operator_manifest: Mapping[str, JSONValue],
) -> bytes:
    return canonical_json_bytes(
        {
            "source_schema_fingerprint": source_fingerprint,
            "variant_schema_fingerprint": variant_fingerprint,
            "operator": operator_manifest,
        }
    )


def _freeze_call_snapshot(
    call: object,
    context: str,
) -> tuple[Mapping[str, JSONValue], str, Mapping[str, JSONValue]]:
    try:
        snapshot = _freeze_mapping(call, context)
    except Exception:
        raise TransformValidationError(f"{context} is invalid") from None
    name = snapshot.get("name")
    arguments = snapshot.get("arguments")
    if type(name) is not str or not isinstance(arguments, Mapping):
        raise TransformValidationError(f"{context} is invalid")
    return snapshot, name, cast(Mapping[str, JSONValue], arguments)


def _call_with_arguments(
    call: Mapping[str, JSONValue],
    arguments: Mapping[str, JSONValue] | dict[str, JSONValue],
    context: str,
) -> Mapping[str, JSONValue]:
    try:
        return _freeze_mapping(
            {key: arguments if key == "arguments" else value for key, value in call.items()},
            context,
        )
    except Exception:
        raise TransformValidationError(f"{context} is invalid") from None


def _surface_outer_to_canonical(
    arguments: Mapping[str, JSONValue],
    plan: _ToolTranslationPlan,
) -> tuple[dict[str, JSONValue], Mapping[str, JSONValue]]:
    canonical_arguments: dict[str, JSONValue] = {}
    container: Mapping[str, JSONValue] | None = None
    outer_required_hits = 0
    for key, value in arguments.items():
        if key == plan.container_name:
            if not isinstance(value, Mapping):
                raise TransformValidationError("parameter restructure surface call is invalid")
            container = cast(Mapping[str, JSONValue], value)
            outer_required_hits += 1
            continue
        if key not in plan.unmoved:
            raise TransformValidationError("parameter restructure surface call is invalid")
        if key in plan.outer_required:
            outer_required_hits += 1
        canonical_arguments[key] = value
    if container is None or outer_required_hits != plan.outer_required_count:
        raise TransformValidationError("parameter restructure surface call is invalid")
    return canonical_arguments, container


def _merge_surface_inner(
    canonical_arguments: dict[str, JSONValue],
    container: Mapping[str, JSONValue],
    plan: _ToolTranslationPlan,
) -> None:
    inner_required_hits = 0
    for key, value in container.items():
        if key not in plan.moved:
            raise TransformValidationError("parameter restructure surface call is invalid")
        if key in plan.inner_required:
            inner_required_hits += 1
        canonical_arguments[key] = value
    if inner_required_hits != plan.inner_required_count:
        raise TransformValidationError("parameter restructure surface call is invalid")


def _surface_snapshot_to_canonical(
    call: Mapping[str, JSONValue],
    arguments: Mapping[str, JSONValue],
    plan: _ToolTranslationPlan,
) -> Mapping[str, JSONValue]:
    canonical_arguments, container = _surface_outer_to_canonical(arguments, plan)
    _merge_surface_inner(canonical_arguments, container, plan)
    return _call_with_arguments(
        call,
        canonical_arguments,
        "parameter restructure canonical call",
    )


def _canonical_snapshot_to_surface(
    call: Mapping[str, JSONValue],
    arguments: Mapping[str, JSONValue],
    plan: _ToolTranslationPlan,
) -> Mapping[str, JSONValue]:
    outer: dict[str, JSONValue] = {}
    inner: dict[str, JSONValue] = {}
    required_hits = 0
    for key, value in arguments.items():
        if key not in plan.declared:
            raise TransformValidationError("parameter restructure canonical call is invalid")
        if key in plan.required:
            required_hits += 1
        if key in plan.moved:
            inner[key] = value
        else:
            outer[key] = value
    if required_hits != plan.required_count:
        raise TransformValidationError("parameter restructure canonical call is invalid")
    outer[plan.container_name] = inner
    return _call_with_arguments(
        call,
        outer,
        "parameter restructure surface call",
    )


def _validated_trace_snapshot(trace: object) -> ExecutionTrace:
    if type(trace) is not ExecutionTrace or not _execution_trace_has_canonical_shape(trace):
        raise TransformValidationError("parameter restructure trace is invalid")
    typed_trace = cast(ExecutionTrace, trace)
    try:
        rebuilt = ExecutionTrace(
            typed_trace.surface_calls,
            typed_trace.semantic_actions,
            typed_trace.base_calls,
        )
    except Exception:
        raise TransformValidationError("parameter restructure trace is invalid") from None
    if rebuilt != typed_trace:
        raise TransformValidationError("parameter restructure trace is invalid")
    return typed_trace


@dataclass(frozen=True, slots=True, eq=False)
class ParameterRestructureTransform:
    """An immutable, schema-attested selective parameter grouping."""

    source_variant: SchemaVariant
    variant: SchemaVariant
    operator: OperatorManifestEntry
    rules: tuple[ParameterGroupRule, ...]
    _source_schema_fingerprint: str = field(init=False, repr=False)
    _variant_schema_fingerprint: str = field(init=False, repr=False)
    _canonical_snapshot: bytes = field(init=False, repr=False)
    _snapshot_fingerprint: str = field(init=False, repr=False)
    _all_tool_names: frozenset[str] = field(init=False, repr=False)
    _changed_tool_names: frozenset[str] = field(init=False, repr=False)
    _plans_by_tool: _TranslationPlanLookup = field(init=False, repr=False)
    _source_runtime_seal: _SchemaRuntimeSeal = field(init=False, repr=False)
    _variant_runtime_seal: _SchemaRuntimeSeal = field(init=False, repr=False)
    _operator_runtime_seal: _OperatorRuntimeSeal = field(init=False, repr=False)
    _plans_root: _MappingRootSeal = field(init=False, repr=False)
    _runtime_rules: tuple[ParameterGroupRule, ...] = field(init=False, repr=False)
    _runtime_all_tool_names: frozenset[str] = field(init=False, repr=False)
    _runtime_changed_tool_names: frozenset[str] = field(init=False, repr=False)
    _runtime_canonical_snapshot: bytes = field(init=False, repr=False)
    _runtime_snapshot_fingerprint: str = field(init=False, repr=False)
    _integrity_token: object = field(init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        validated = _validate_construction(
            self.source_variant,
            self.variant,
            self.operator,
            self.rules,
        )
        canonical = _transform_snapshot(
            validated.source_fingerprint,
            validated.variant_fingerprint,
            validated.operator_manifest,
        )
        all_tool_names, changed_tool_names, plans_by_tool = _build_translation_plans(
            self.source_variant,
            validated.rules,
        )
        snapshot_fingerprint = hashlib.sha256(canonical).hexdigest()
        object.__setattr__(self, "rules", validated.rules)
        object.__setattr__(
            self,
            "_source_schema_fingerprint",
            validated.source_fingerprint,
        )
        object.__setattr__(
            self,
            "_variant_schema_fingerprint",
            validated.variant_fingerprint,
        )
        object.__setattr__(self, "_canonical_snapshot", canonical)
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            snapshot_fingerprint,
        )
        object.__setattr__(self, "_all_tool_names", all_tool_names)
        object.__setattr__(self, "_changed_tool_names", changed_tool_names)
        object.__setattr__(self, "_plans_by_tool", plans_by_tool)
        object.__setattr__(
            self,
            "_source_runtime_seal",
            _make_schema_runtime_seal(
                self.source_variant,
                validated.source_fingerprint,
            ),
        )
        object.__setattr__(
            self,
            "_variant_runtime_seal",
            _make_schema_runtime_seal(
                self.variant,
                validated.variant_fingerprint,
            ),
        )
        object.__setattr__(
            self,
            "_operator_runtime_seal",
            _make_operator_runtime_seal(self.operator),
        )
        object.__setattr__(
            self,
            "_plans_root",
            _make_mapping_root_seal(
                cast(Mapping[str, JSONValue], plans_by_tool),
            ),
        )
        object.__setattr__(self, "_runtime_rules", validated.rules)
        object.__setattr__(self, "_runtime_all_tool_names", all_tool_names)
        object.__setattr__(self, "_runtime_changed_tool_names", changed_tool_names)
        object.__setattr__(self, "_runtime_canonical_snapshot", canonical)
        object.__setattr__(
            self,
            "_runtime_snapshot_fingerprint",
            snapshot_fingerprint,
        )
        object.__setattr__(self, "_integrity_token", _PARAMETER_RESTRUCTURE_INTEGRITY_TOKEN)

    def _require_integrity(self) -> None:
        try:
            if (
                object.__getattribute__(self, "_integrity_token")
                is not _PARAMETER_RESTRUCTURE_INTEGRITY_TOKEN
                or object.__getattribute__(self, "source_variant")
                is not self._source_runtime_seal.variant
                or object.__getattribute__(self, "variant")
                is not self._variant_runtime_seal.variant
                or object.__getattribute__(self, "operator")
                is not self._operator_runtime_seal.operator
                or object.__getattribute__(self, "rules") is not self._runtime_rules
                or object.__getattribute__(self, "_all_tool_names")
                is not self._runtime_all_tool_names
                or object.__getattribute__(self, "_changed_tool_names")
                is not self._runtime_changed_tool_names
                or object.__getattribute__(self, "_plans_by_tool") is not self._plans_root.mapping
                or object.__getattribute__(self, "_source_schema_fingerprint")
                != self._source_runtime_seal.fingerprint
                or object.__getattribute__(self, "_variant_schema_fingerprint")
                != self._variant_runtime_seal.fingerprint
                or object.__getattribute__(self, "_canonical_snapshot")
                is not self._runtime_canonical_snapshot
                or object.__getattribute__(self, "_snapshot_fingerprint")
                != self._runtime_snapshot_fingerprint
                or not _schema_runtime_seal_matches(self._source_runtime_seal)
                or not _schema_runtime_seal_matches(self._variant_runtime_seal)
                or not _operator_runtime_seal_matches(self._operator_runtime_seal)
                or not _mapping_root_seal_matches(self._plans_root)
            ):
                raise TransformValidationError(
                    "parameter restructure transform integrity validation failed"
                )
        except Exception:
            raise TransformValidationError(
                "parameter restructure transform integrity validation failed"
            ) from None

    def _plan_for_name(self, name: str) -> _ToolTranslationPlan | None:
        if name not in self._all_tool_names:
            raise TransformValidationError("parameter restructure call has an unknown tool")
        if name not in self._changed_tool_names:
            return None
        try:
            plan = self._plans_by_tool[name]
            plan.require_integrity()
            return plan
        except Exception:
            raise TransformValidationError(
                "parameter restructure transform integrity validation failed"
            ) from None

    def surface_call_to_canonical(
        self,
        call: Mapping[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        """Translate one nested surface call into the flat source interface."""

        self._require_integrity()
        snapshot, name, arguments = _freeze_call_snapshot(
            call,
            "parameter restructure surface call",
        )
        plan = self._plan_for_name(name)
        if plan is None:
            return snapshot
        return _surface_snapshot_to_canonical(snapshot, arguments, plan)

    def canonical_call_to_surface(
        self,
        call: Mapping[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        """Translate one flat source call into the nested final interface."""

        self._require_integrity()
        snapshot, name, arguments = _freeze_call_snapshot(
            call,
            "parameter restructure canonical call",
        )
        plan = self._plan_for_name(name)
        if plan is None:
            return snapshot
        return _canonical_snapshot_to_surface(snapshot, arguments, plan)

    def _trace_for_source(self, trace: ExecutionTrace) -> ExecutionTrace:
        self._require_integrity()
        try:
            validated = _validated_trace_snapshot(trace)
            translated_calls: list[Mapping[str, JSONValue]] = []
            saw_canonical = False
            saw_surface = False
            for call in validated.surface_calls:
                snapshot, name, arguments = _freeze_call_snapshot(
                    call,
                    "parameter restructure trace surface call",
                )
                plan = self._plan_for_name(name)
                if plan is None:
                    translated_calls.append(snapshot)
                    continue
                if plan.container_name in arguments:
                    if any(key in plan.moved for key in arguments):
                        raise TransformValidationError(
                            "parameter restructure trace contains a hybrid call"
                        )
                    saw_surface = True
                    translated_calls.append(
                        _surface_snapshot_to_canonical(snapshot, arguments, plan)
                    )
                else:
                    saw_canonical = True
                    _canonical_snapshot_to_surface(snapshot, arguments, plan)
                    translated_calls.append(snapshot)
                if saw_canonical and saw_surface:
                    raise TransformValidationError(
                        "parameter restructure trace mixes canonical and surface modes"
                    )
            if not saw_surface:
                return validated
            return ExecutionTrace(
                tuple(translated_calls),
                validated.semantic_actions,
                validated.base_calls,
            )
        except TransformValidationError:
            raise
        except Exception:
            raise TransformValidationError("parameter restructure trace is invalid") from None


def build_parameter_restructure_transform(
    base_variant: SchemaVariant,
    *,
    rules: tuple[ParameterGroupRule, ...],
    seed: int,
) -> ParameterRestructureTransform:
    """Build one deterministic flat-to-nested parameter schema transform."""

    normalized = _normalize_rules(rules)
    if type(base_variant) is not SchemaVariant:
        raise TransformValidationError("base_variant must be a valid SchemaVariant")
    try:
        base_fingerprint = schema_fingerprint(base_variant)
    except Exception:
        raise TransformValidationError("base_variant must be a valid SchemaVariant") from None
    if base_variant.manifest.get("kind") == "toolshift_interface_variant":
        raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
    tools = _build_restructured_tools(base_variant, normalized)
    parameters = _operator_parameters(normalized)
    operator = OperatorManifestEntry(
        _PARAMETER_RESTRUCTURE_OPERATOR_ID,
        _PARAMETER_RESTRUCTURE_OPERATOR,
        _PARAMETER_RESTRUCTURE_LEVEL,
        derive_operator_seed(
            seed,
            base_fingerprint,
            _PARAMETER_RESTRUCTURE_OPERATOR_ID,
            _PARAMETER_RESTRUCTURE_OPERATOR,
            0,
        ),
        PARAMETER_RESTRUCTURE_VERSION_HASH,
        parameters,
    )
    variant = build_transformed_variant(
        base_variant,
        tools,
        seed=seed,
        operators=(operator,),
    )
    return ParameterRestructureTransform(
        base_variant,
        variant,
        operator,
        normalized,
    )


class ParameterRestructureAdapter(SemanticAdapter):
    """Delegate one nested parameter interface to an exact flat source adapter."""

    __slots__ = (
        "_final_schema_fingerprint",
        "_final_variant",
        "_source_adapter",
        "_source_adapter_seal",
        "_source_schema_fingerprint",
        "_source_variant",
        "_transform",
        "_transform_seal",
    )

    def __init__(
        self,
        source_adapter: SemanticAdapter,
        transform: ParameterRestructureTransform,
    ) -> None:
        if not isinstance(source_adapter, SemanticAdapter):
            raise TransformValidationError("source_adapter must be a SemanticAdapter")
        if isinstance(source_adapter, ParameterRestructureAdapter):
            raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
        if type(transform) is not ParameterRestructureTransform:
            raise TransformValidationError("transform must be a ParameterRestructureTransform")
        transform._require_integrity()
        source_variant = transform.source_variant
        if source_variant.manifest.get("kind") == "toolshift_interface_variant":
            raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
        try:
            source_public = source_adapter.variant
            source_raw = _raw_adapter_variant(source_adapter)
        except Exception:
            raise TransformValidationError("source adapter binding validation failed") from None
        if (
            type(source_public) is SchemaVariant
            and source_public.manifest.get("kind") == "toolshift_interface_variant"
        ) or (
            type(source_raw) is SchemaVariant
            and source_raw.manifest.get("kind") == "toolshift_interface_variant"
        ):
            raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
        if source_public is not source_variant or source_raw is not source_variant:
            raise TransformValidationError("source adapter binding does not match transform")
        try:
            source_fingerprint = schema_fingerprint(source_variant)
            public_fingerprint = schema_fingerprint(source_public)
            raw_fingerprint = schema_fingerprint(source_raw)
            final_fingerprint = schema_fingerprint(transform.variant)
        except Exception:
            raise TransformValidationError("source adapter binding validation failed") from None
        if public_fingerprint != source_fingerprint or raw_fingerprint != source_fingerprint:
            raise TransformValidationError("source adapter binding does not match transform")
        super().__init__(transform.variant)
        self._source_adapter = source_adapter
        self._source_adapter_seal = source_adapter
        self._transform = transform
        self._transform_seal = transform
        self._source_variant = source_variant
        self._source_schema_fingerprint = source_fingerprint
        self._final_variant = transform.variant
        self._final_schema_fingerprint = final_fingerprint
        self._require_bindings()

    @property
    def transform(self) -> ParameterRestructureTransform:
        """The immutable parameter restructuring handled by this adapter."""

        return self._transform

    def _require_bindings(self) -> None:
        try:
            if (
                self._transform is not self._transform_seal
                or self._source_adapter is not self._source_adapter_seal
                or self._transform.source_variant is not self._source_variant
                or self._transform.variant is not self._final_variant
                or self._source_schema_fingerprint
                != self._transform._source_runtime_seal.fingerprint
                or self._final_schema_fingerprint
                != self._transform._variant_runtime_seal.fingerprint
            ):
                raise TransformValidationError("adapter binding integrity validation failed")
            self._transform._require_integrity()
        except Exception:
            raise TransformValidationError("adapter binding integrity validation failed") from None
        try:
            source_public = self._source_adapter.variant
            source_raw = _raw_adapter_variant(self._source_adapter)
            final_public = self.variant
            final_raw = _raw_adapter_variant(self)
        except Exception:
            raise TransformValidationError("adapter binding integrity validation failed") from None
        if (
            source_public is not self._source_variant
            or source_raw is not self._source_variant
            or final_public is not self._final_variant
            or final_raw is not self._final_variant
        ):
            raise TransformValidationError("adapter binding integrity validation failed")

    def _delegate(
        self,
        operation: Callable[[], _ResultT],
        expected_error: str,
    ) -> _ResultT:
        return _delegate_with_binding_guard(
            self._require_bindings,
            operation,
            expected_error,
        )

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        canonical_call = self._transform.surface_call_to_canonical(surface_call)
        return self._delegate(
            lambda: self._source_adapter.surface_to_semantic(canonical_call),
            "source adapter rejected the translated surface call",
        )

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        return self._delegate(
            lambda: self._source_adapter.semantic_to_base_calls(action),
            "source adapter rejected the semantic action",
        )

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        canonical_call = self._transform.surface_call_to_canonical(surface_call)
        return self._delegate(
            lambda: self._source_adapter.base_observation_to_surface(
                canonical_call,
                actions,
                base_observation_groups,
            ),
            "source adapter rejected observation wrapping",
        )

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        canonical_trace = self._transform._trace_for_source(trace)
        return self._delegate(
            lambda: self._source_adapter.canonicalize_trace(canonical_trace),
            "source adapter rejected trace canonicalization",
        )


def apply_parameter_restructure(
    source_adapter: SemanticAdapter,
    *,
    rules: tuple[ParameterGroupRule, ...],
    seed: int,
) -> ParameterRestructureAdapter:
    """Build and bind one parameter restructuring to an exact source adapter."""

    if not isinstance(source_adapter, SemanticAdapter):
        raise TransformValidationError("source_adapter must be a SemanticAdapter")
    if isinstance(source_adapter, ParameterRestructureAdapter):
        raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
    try:
        source_variant = _raw_adapter_variant(source_adapter)
    except Exception:
        raise TransformValidationError("source adapter binding validation failed") from None
    if type(source_variant) is not SchemaVariant:
        raise TransformValidationError("source adapter binding validation failed")
    if source_variant.manifest.get("kind") == "toolshift_interface_variant":
        raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
    transform = build_parameter_restructure_transform(
        source_variant,
        rules=rules,
        seed=seed,
    )
    return ParameterRestructureAdapter(source_adapter, transform)


__all__ = [
    "PARAMETER_RESTRUCTURE_VERSION_HASH",
    "ParameterGroupRule",
    "ParameterRestructureAdapter",
    "ParameterRestructureTransform",
    "apply_parameter_restructure",
    "build_parameter_restructure_transform",
]
