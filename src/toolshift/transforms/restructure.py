"""Deterministic L2 grouping of selected top-level tool parameters."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from toolshift.contracts.schema import schema_fingerprint
from toolshift.transforms.base import (
    OperatorManifestEntry,
    TransformValidationError,
    build_transformed_variant,
    derive_operator_seed,
)
from toolshift.types import (
    JSONValue,
    SchemaVariant,
    SurfaceToolSpec,
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
        if isinstance(child, (list, tuple)):
            return True
        if _schema_node_contains_forbidden_keyword(child):
            return True
    return False


def _array_subschemas_contain_forbidden_keyword(value: Mapping[object, object]) -> bool:
    return any(
        isinstance(children, (list, tuple))
        and any(_schema_node_contains_forbidden_keyword(child) for child in children)
        for children in (value.get(key) for key in _ARRAY_SUBSCHEMA_KEYS)
    )


def _mapping_subschemas_contain_forbidden_keyword(value: Mapping[object, object]) -> bool:
    return any(
        isinstance(children, Mapping)
        and any(_schema_node_contains_forbidden_keyword(child) for child in children.values())
        for children in (value.get(key) for key in _MAPPING_SUBSCHEMA_KEYS)
    )


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
    _build_restructured_tools(typed_source, normalized)
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
            hashlib.sha256(canonical).hexdigest(),
        )


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


__all__ = [
    "PARAMETER_RESTRUCTURE_VERSION_HASH",
    "ParameterGroupRule",
    "ParameterRestructureTransform",
    "build_parameter_restructure_transform",
]
