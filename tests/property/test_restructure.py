"""Schema and manifest tests for deterministic L2 parameter grouping."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

import toolshift.transforms.rename as rename_module
import toolshift.transforms.restructure as restructure_module
from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.schema import schema_fingerprint
from toolshift.transforms.base import (
    OperatorManifestEntry,
    TransformValidationError,
    build_transformed_variant,
    derive_operator_seed,
)
from toolshift.transforms.restructure import (
    PARAMETER_RESTRUCTURE_VERSION_HASH,
    ParameterGroupRule,
    ParameterRestructureTransform,
    build_parameter_restructure_transform,
)
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    canonical_json_bytes,
    manifest_sha256,
)


def _search_schema() -> dict[str, JSONValue]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Search input",
        "description": "Arguments accepted by search",
        "$comment": "runtime fixture",
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 0},
            "locale": {"type": ("string", "null")},
        },
        "required": ("query", "locale"),
        "additionalProperties": False,
    }


def _summarize_schema() -> dict[str, JSONValue]:
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "style": {"enum": ("short", "long")},
        },
        "required": ("text",),
        "additionalProperties": False,
    }


def _base_variant(
    *,
    search_schema: Mapping[str, JSONValue] | None = None,
    manifest: Mapping[str, JSONValue] | None = None,
) -> SchemaVariant:
    return SchemaVariant(
        "base-v1",
        (
            SurfaceToolSpec(
                "search",
                "Search records",
                search_schema if search_schema is not None else _search_schema(),
            ),
            SurfaceToolSpec("summarize", "Summarize records", _summarize_schema()),
            SurfaceToolSpec("status", "Read status", {"type": "string"}),
        ),
        manifest if manifest is not None else {"operator": "identity", "seed": 3},
    )


def _tool(variant: SchemaVariant, name: str) -> SurfaceToolSpec:
    return next(tool for tool in variant.tools if tool.name == name)


def _properties(tool: SurfaceToolSpec) -> Mapping[str, JSONValue]:
    properties = tool.input_schema["properties"]
    assert isinstance(properties, Mapping)
    return cast(Mapping[str, JSONValue], properties)


def _search_rule(
    moved_parameters: tuple[str, ...] = ("query", "limit"),
    *,
    container_name: str = "request",
) -> ParameterGroupRule:
    return ParameterGroupRule("search", container_name, moved_parameters)


class _RecordingRestructureSource(SemanticAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._tool_names = frozenset(tool.name for tool in variant.tools)
        self.parse_inputs: list[Mapping[str, JSONValue]] = []
        self.parse_results: list[tuple[SemanticAction, ...]] = []
        self.compile_inputs: list[SemanticAction] = []
        self.compile_results: list[tuple[Mapping[str, JSONValue], ...]] = []
        self.wrap_inputs: list[
            tuple[
                Mapping[str, JSONValue],
                tuple[SemanticAction, ...],
                tuple[tuple[JSONValue, ...], ...],
            ]
        ] = []
        self.trace_inputs: list[ExecutionTrace] = []

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        self.parse_inputs.append(surface_call)
        name = surface_call.get("name")
        arguments = surface_call.get("arguments")
        if type(name) is not str or name not in self._tool_names:
            raise ValueError("PRIVATE_PARSE_PAYLOAD")
        if not isinstance(arguments, Mapping):
            raise ValueError("PRIVATE_PARSE_PAYLOAD")
        result = (SemanticAction(name, arguments),)
        self.parse_results.append(result)
        return result

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self.compile_inputs.append(action)
        result = ({"name": action.name, "arguments": action.arguments},)
        self.compile_results.append(result)
        return result

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self.wrap_inputs.append((surface_call, actions, base_observation_groups))
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self.trace_inputs.append(trace)
        return trace.semantic_actions


class _IdentityArgumentShapeSource(_RecordingRestructureSource):
    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        self.parse_inputs.append(surface_call)
        result = (SemanticAction("status", {}),)
        self.parse_results.append(result)
        return result

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self.wrap_inputs.append((surface_call, actions, base_observation_groups))
        return base_observation_groups[0][0]


class _ExplodingManifestMapping(Mapping[str, JSONValue]):
    def __getitem__(self, key: str) -> JSONValue:
        raise RuntimeError("PRIVATE_MANIFEST_PAYLOAD")

    def __iter__(self):
        return iter(("kind",))

    def __len__(self) -> int:
        return 1


class _ManifestPayloadPublicSource(_RecordingRestructureSource):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        public_variant = _base_variant()
        object.__setattr__(public_variant, "manifest", _ExplodingManifestMapping())
        self._public_variant = public_variant

    @property
    def variant(self) -> SchemaVariant:
        return self._public_variant


class _ExplodingCallMapping(dict[str, JSONValue]):
    def items(self):
        raise RuntimeError("PRIVATE_CALLBACK_PAYLOAD")


def _apply_restructure(
    source: SemanticAdapter | None = None,
    *,
    rules: tuple[ParameterGroupRule, ...] | None = None,
):
    source_adapter = source or _RecordingRestructureSource(_base_variant())
    return restructure_module.apply_parameter_restructure(
        source_adapter,
        rules=rules or (_search_rule(),),
        seed=7,
    )


def test_parameter_restructure_abi_hash_has_fixed_vector() -> None:
    assert (
        PARAMETER_RESTRUCTURE_VERSION_HASH
        == "6a170c4eb544cda3798a4106d30625e9cf25b46878fbea7e27589895348d45ac"
    )


def test_restructure_module_exports_adapter_and_apply_helper() -> None:
    assert restructure_module.__all__ == [
        "PARAMETER_RESTRUCTURE_VERSION_HASH",
        "ParameterGroupRule",
        "ParameterRestructureAdapter",
        "ParameterRestructureTransform",
        "apply_parameter_restructure",
        "build_parameter_restructure_transform",
    ]


def test_parameter_group_rule_is_normalized_frozen_unhashable_value() -> None:
    rule = _search_rule(("query", "limit"))
    equivalent = _search_rule(("limit", "query"))

    assert rule.tool_name == "search"
    assert rule.container_name == "request"
    assert rule.moved_parameters == ("limit", "query")
    assert rule == equivalent
    assert rule is not equivalent
    assert not hasattr(rule, "__dict__")
    with pytest.raises(FrozenInstanceError):
        rule.tool_name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(rule)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tool_name", ""),
        ("tool_name", "  "),
        ("tool_name", 1),
        ("tool_name", "\ud800"),
        ("container_name", ""),
        ("container_name", "  "),
        ("container_name", False),
        ("container_name", "\udfff"),
    ],
)
def test_parameter_group_rule_rejects_invalid_names(field_name: str, value: object) -> None:
    values: dict[str, object] = {
        "tool_name": "search",
        "container_name": "request",
        "moved_parameters": ("query",),
    }
    values[field_name] = value

    with pytest.raises(TransformValidationError):
        ParameterGroupRule(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "moved_parameters",
    [
        [],
        (),
        ("query", "query"),
        ("query", 1),
        ("",),
        ("  ",),
        ("\ud800",),
    ],
)
def test_parameter_group_rule_rejects_invalid_moved_parameters(
    moved_parameters: object,
) -> None:
    with pytest.raises(TransformValidationError):
        ParameterGroupRule("search", "request", moved_parameters)  # type: ignore[arg-type]


def test_rule_and_rule_sequence_order_do_not_change_manifest_or_variant() -> None:
    base = _base_variant()
    first = build_parameter_restructure_transform(
        base,
        rules=(
            ParameterGroupRule("summarize", "payload", ("style", "text")),
            _search_rule(("query", "limit")),
        ),
        seed=7,
    )
    second = build_parameter_restructure_transform(
        base,
        rules=(
            _search_rule(("limit", "query")),
            ParameterGroupRule("summarize", "payload", ("text", "style")),
        ),
        seed=7,
    )

    assert tuple(rule.tool_name for rule in first.rules) == ("search", "summarize")
    assert first.rules == second.rules
    assert first.operator.parameters == second.operator.parameters
    assert first.operator.seed == second.operator.seed
    assert first.variant.variant_id == second.variant.variant_id
    assert first.variant == second.variant


def test_operator_manifest_has_exact_normalized_path_segment_shape() -> None:
    base = _base_variant()
    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(("query", "limit")),),
        seed=7,
    )
    expected_parameters = {
        "mode": "nest_top_level",
        "rules": (
            {
                "tool_name": "search",
                "container_name": "request",
                "moves": (
                    {
                        "source_path": ("limit",),
                        "surface_path": ("request", "limit"),
                    },
                    {
                        "source_path": ("query",),
                        "surface_path": ("request", "query"),
                    },
                ),
            },
        ),
    }

    assert transform.operator.operator_id == "restructure-000"
    assert transform.operator.operator == "parameter_restructure"
    assert transform.operator.level == "L2"
    assert transform.operator.version_hash == PARAMETER_RESTRUCTURE_VERSION_HASH
    assert transform.operator.seed == derive_operator_seed(
        7,
        schema_fingerprint(base),
        "restructure-000",
        "parameter_restructure",
        0,
    )
    assert transform.operator.parameters == expected_parameters
    operators = transform.variant.manifest["operators"]
    assert isinstance(operators, tuple)
    operator_manifest = operators[0]
    assert isinstance(operator_manifest, Mapping)
    assert operator_manifest["parameters"] == expected_parameters


def test_seed_changes_operator_manifest_and_variant_id() -> None:
    base = _base_variant()
    rules = (_search_rule(),)
    first = build_parameter_restructure_transform(base, rules=rules, seed=7)
    second = build_parameter_restructure_transform(base, rules=rules, seed=8)

    assert first.operator.seed != second.operator.seed
    assert first.operator.as_manifest() != second.operator.as_manifest()
    assert first.variant.manifest != second.variant.manifest
    assert first.variant.variant_id != second.variant.variant_id


def test_selective_grouping_builds_exact_closed_required_schema() -> None:
    base = _base_variant()
    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )
    source = _tool(base, "search")
    source_properties = _properties(source)
    target = _tool(transform.variant, "search")

    assert target is not source
    assert target.name == source.name
    assert target.description == source.description
    assert target.input_schema == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Search input",
        "description": "Arguments accepted by search",
        "$comment": "runtime fixture",
        "type": "object",
        "properties": {
            "locale": source_properties["locale"],
            "request": {
                "type": "object",
                "properties": {
                    "limit": source_properties["limit"],
                    "query": source_properties["query"],
                },
                "required": ("query",),
                "additionalProperties": False,
            },
        },
        "required": ("locale", "request"),
        "additionalProperties": False,
    }


def test_optional_only_whole_wrap_still_requires_an_empty_capable_container() -> None:
    schema: dict[str, JSONValue] = {
        "type": "object",
        "properties": {
            "alpha": {"type": "string"},
            "beta": {"type": "integer"},
        },
        "additionalProperties": False,
    }
    base = _base_variant(search_schema=schema)
    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(("beta", "alpha")),),
        seed=7,
    )
    source_properties = _properties(_tool(base, "search"))
    target_schema = _tool(transform.variant, "search").input_schema

    assert target_schema == {
        "type": "object",
        "properties": {
            "request": {
                "type": "object",
                "properties": {
                    "alpha": source_properties["alpha"],
                    "beta": source_properties["beta"],
                },
                "additionalProperties": False,
            },
        },
        "required": ("request",),
        "additionalProperties": False,
    }


def test_boolean_property_subschema_is_moved_without_reinterpretation() -> None:
    schema: dict[str, JSONValue] = {
        "type": "object",
        "properties": {"flag": True, "note": {"type": "string"}},
        "additionalProperties": False,
    }
    transform = build_parameter_restructure_transform(
        _base_variant(search_schema=schema),
        rules=(_search_rule(("flag",)),),
        seed=7,
    )
    outer_properties = _properties(_tool(transform.variant, "search"))
    container = outer_properties["request"]
    assert isinstance(container, Mapping)
    inner_properties = container["properties"]
    assert isinstance(inner_properties, Mapping)
    assert inner_properties["flag"] is True


def test_identity_tools_preserve_exact_surface_tool_objects() -> None:
    base = _base_variant()
    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )

    assert _tool(transform.variant, "summarize") is _tool(base, "summarize")
    assert _tool(transform.variant, "status") is _tool(base, "status")


def test_builder_does_not_modify_base_schema_or_caller_rules() -> None:
    raw_schema = _search_schema()
    raw_snapshot = copy.deepcopy(raw_schema)
    base = _base_variant(search_schema=raw_schema)
    base_fingerprint = schema_fingerprint(base)
    base_tools = base.tools
    rule = _search_rule(("query", "limit"))
    rules = (rule,)
    rule_snapshot = (rule.tool_name, rule.container_name, rule.moved_parameters)

    transform = build_parameter_restructure_transform(base, rules=rules, seed=7)

    assert raw_schema == raw_snapshot
    assert schema_fingerprint(base) == base_fingerprint
    assert base.tools is base_tools
    assert rules == (rule,)
    assert (rule.tool_name, rule.container_name, rule.moved_parameters) == rule_snapshot
    assert transform.rules[0] is not rule


@pytest.mark.parametrize("rules", [[], (), (object(),)])
def test_builder_requires_an_exact_nonempty_rule_tuple(rules: object) -> None:
    with pytest.raises(TransformValidationError):
        build_parameter_restructure_transform(
            _base_variant(),
            rules=rules,  # type: ignore[arg-type]
            seed=7,
        )


def test_builder_rejects_tuple_subclass_for_rules() -> None:
    class RuleTuple(tuple[ParameterGroupRule, ...]):
        pass

    with pytest.raises(TransformValidationError):
        build_parameter_restructure_transform(
            _base_variant(),
            rules=RuleTuple((_search_rule(),)),  # type: ignore[arg-type]
            seed=7,
        )


def test_builder_rejects_duplicate_rules_for_one_tool() -> None:
    with pytest.raises(TransformValidationError, match="duplicate"):
        build_parameter_restructure_transform(
            _base_variant(),
            rules=(_search_rule(("query",)), _search_rule(("limit",))),
            seed=7,
        )


def test_builder_rejects_unknown_tool_without_payload_disclosure() -> None:
    rule = ParameterGroupRule("PRIVATE_TOOL", "request", ("query",))

    with pytest.raises(TransformValidationError) as caught:
        build_parameter_restructure_transform(_base_variant(), rules=(rule,), seed=7)

    assert "PRIVATE_TOOL" not in str(caught.value)


def test_builder_rejects_unknown_moved_parameter_without_payload_disclosure() -> None:
    rule = _search_rule(("PRIVATE_PARAMETER",))

    with pytest.raises(TransformValidationError) as caught:
        build_parameter_restructure_transform(_base_variant(), rules=(rule,), seed=7)

    assert "PRIVATE_PARAMETER" not in str(caught.value)


@pytest.mark.parametrize("container_name", ["locale", "query"])
def test_builder_requires_container_fresh_against_every_source_property(
    container_name: str,
) -> None:
    with pytest.raises(TransformValidationError, match="container"):
        build_parameter_restructure_transform(
            _base_variant(),
            rules=(_search_rule(("query",), container_name=container_name),),
            seed=7,
        )


def test_identity_tool_with_unsupported_schema_is_allowed_until_selected() -> None:
    base = _base_variant()
    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )
    assert _tool(transform.variant, "status") is _tool(base, "status")

    with pytest.raises(TransformValidationError, match="schema"):
        build_parameter_restructure_transform(
            base,
            rules=(ParameterGroupRule("status", "payload", ("value",)),),
            seed=7,
        )


@pytest.mark.parametrize(
    "case",
    [
        "missing_type",
        "wrong_type",
        "missing_properties",
        "empty_properties",
        "non_mapping_properties",
        "missing_additional_properties",
        "open_object",
        "root_combinator",
        "root_conditional",
        "root_dependencies",
        "root_pattern_properties",
        "root_property_names",
        "root_unevaluated_properties",
        "root_min_properties",
        "root_max_properties",
        "root_definitions",
        "root_extension",
        "false_like_additional_properties",
    ],
)
def test_builder_rejects_unsupported_root_schema_grammar(case: str) -> None:
    schema = _search_schema()
    if case == "missing_type":
        schema.pop("type")
    elif case == "wrong_type":
        schema["type"] = ("object",)
    elif case == "missing_properties":
        schema.pop("properties")
    elif case == "empty_properties":
        schema["properties"] = {}
    elif case == "non_mapping_properties":
        schema["properties"] = ("query",)
    elif case == "missing_additional_properties":
        schema.pop("additionalProperties")
    elif case == "open_object":
        schema["additionalProperties"] = True
    elif case == "root_combinator":
        schema["allOf"] = ()
    elif case == "root_conditional":
        schema["if"] = {"required": ("query",)}
    elif case == "root_dependencies":
        schema["dependentRequired"] = {"query": ("locale",)}
    elif case == "root_pattern_properties":
        schema["patternProperties"] = {"^x": {"type": "string"}}
    elif case == "root_property_names":
        schema["propertyNames"] = {"minLength": 1}
    elif case == "root_unevaluated_properties":
        schema["unevaluatedProperties"] = False
    elif case == "root_min_properties":
        schema["minProperties"] = 1
    elif case == "root_max_properties":
        schema["maxProperties"] = 3
    elif case == "root_definitions":
        schema["$defs"] = {"value": {"type": "string"}}
    elif case == "root_extension":
        schema["x-private"] = "PRIVATE"
    else:
        schema["additionalProperties"] = 0

    with pytest.raises(TransformValidationError, match="schema"):
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )


@pytest.mark.parametrize(
    "required",
    [
        None,
        "query",
        ("query", "query"),
        ("missing",),
        ("query", 1),
    ],
)
def test_builder_rejects_invalid_required_declarations(required: JSONValue) -> None:
    schema = _search_schema()
    schema["required"] = required

    with pytest.raises(TransformValidationError, match="schema"):
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )


def test_builder_rejects_blank_property_name() -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties[""] = {"type": "string"}

    with pytest.raises(TransformValidationError, match="schema"):
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )


def test_builder_rejects_non_schema_property_value() -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties["query"] = 7

    with pytest.raises(TransformValidationError, match="schema"):
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )


@pytest.mark.parametrize(
    ("keyword", "bad_value"),
    [
        ("items", "PRIVATE_SINGLE"),
        ("additionalProperties", 7),
        ("not", None),
        ("contains", 1.5),
        ("contentSchema", "PRIVATE_CONTENT"),
        ("else", "PRIVATE_ELSE"),
        ("if", "PRIVATE_IF"),
        ("propertyNames", [{"type": "string"}]),
        ("then", ({"type": "string"},)),
        ("unevaluatedItems", "PRIVATE_UNEVALUATED_ITEMS"),
        ("unevaluatedProperties", ({"type": "string"},)),
    ],
)
def test_builder_rejects_invalid_single_schema_keyword_shape(
    keyword: str,
    bad_value: JSONValue,
) -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties["query"] = {"type": "string", keyword: bad_value}

    with pytest.raises(TransformValidationError) as caught:
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )

    assert str(caught.value) == "parameter restructure source schema is unsupported"


@pytest.mark.parametrize(
    ("keyword", "bad_value"),
    [
        ("allOf", {"title": "PRIVATE_CONTAINER", "type": "string"}),
        ("anyOf", "PRIVATE_ARRAY"),
        ("oneOf", ({"type": "string"}, "PRIVATE_CHILD")),
        ("prefixItems", [True, "PRIVATE_CHILD"]),
    ],
)
def test_builder_rejects_invalid_schema_array_keyword_shape(
    keyword: str,
    bad_value: JSONValue,
) -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties["query"] = {"type": "string", keyword: bad_value}

    with pytest.raises(TransformValidationError) as caught:
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )

    assert str(caught.value) == "parameter restructure source schema is unsupported"


@pytest.mark.parametrize(
    ("keyword", "bad_value"),
    [
        ("properties", "PRIVATE_MAPPING"),
        ("$defs", ({"const": "PRIVATE_CONTAINER"},)),
        ("definitions", {"x": "PRIVATE_CHILD"}),
        ("patternProperties", {"^x$": "PRIVATE_CHILD"}),
        ("dependentSchemas", {"x": "PRIVATE_CHILD"}),
    ],
)
def test_builder_rejects_invalid_schema_mapping_keyword_shape(
    keyword: str,
    bad_value: JSONValue,
) -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties["query"] = {"type": "string", keyword: bad_value}

    with pytest.raises(TransformValidationError) as caught:
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )

    assert str(caught.value) == "parameter restructure source schema is unsupported"


def test_builder_accepts_valid_nested_schema_keyword_shapes() -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    query_schema: dict[str, JSONValue] = {
        "type": "object",
        "propertyNames": True,
        "additionalProperties": {"type": "string"},
        "allOf": [True, {"type": "object"}],
        "properties": {
            "flag": False,
            "count": {"type": "integer"},
        },
    }
    properties["query"] = query_schema

    transform = build_parameter_restructure_transform(
        _base_variant(search_schema=schema),
        rules=(_search_rule(("query",)),),
        seed=7,
    )

    request = _properties(_tool(transform.variant, "search"))["request"]
    assert isinstance(request, Mapping)
    moved = request["properties"]
    assert isinstance(moved, Mapping)
    source_query = _properties(_tool(transform.source_variant, "search"))["query"]
    assert moved["query"] == source_query


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("$ref", "#/$defs/private"),
        ("$dynamicRef", "#private"),
        ("$recursiveRef", "#"),
        ("$recursiveAnchor", True),
        ("$id", "private"),
        ("$anchor", "private"),
        ("$dynamicAnchor", "private"),
        ("default", "PRIVATE"),
        ("examples", ("PRIVATE",)),
    ],
)
def test_builder_recursively_rejects_position_or_default_sensitive_keywords(
    keyword: str,
    value: JSONValue,
) -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties["query"] = {
        "type": "object",
        "properties": {"nested": {"type": "string", keyword: value}},
    }

    with pytest.raises(TransformValidationError) as caught:
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )

    assert "PRIVATE" not in str(caught.value)


def test_builder_rejects_non_draft_2020_schema_uri_without_payload_disclosure() -> None:
    schema = _search_schema()
    schema["$schema"] = "PRIVATE_DIALECT"

    with pytest.raises(TransformValidationError) as caught:
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )

    assert "PRIVATE_DIALECT" not in str(caught.value)


def test_builder_rejects_nested_non_draft_2020_schema_without_payload_disclosure() -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties["query"] = {
        "type": "array",
        "items": {
            "$schema": "PRIVATE_DIALECT",
            "type": "string",
        },
    }

    with pytest.raises(TransformValidationError) as caught:
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )

    assert "PRIVATE_DIALECT" not in str(caught.value)


def test_builder_accepts_nested_canonical_draft_2020_schema() -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    nested: dict[str, JSONValue] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "string",
    }
    properties["query"] = {"type": "array", "items": nested}

    transform = build_parameter_restructure_transform(
        _base_variant(search_schema=schema),
        rules=(_search_rule(("query",)),),
        seed=7,
    )

    request = _properties(_tool(transform.variant, "search"))["request"]
    assert isinstance(request, Mapping)
    moved = request["properties"]
    assert isinstance(moved, Mapping)
    assert moved["query"] == properties["query"]


def test_whitespace_only_property_name_is_preserved_when_unmoved() -> None:
    schema: dict[str, JSONValue] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            " ": {"type": "integer"},
        },
        "required": ("query", " "),
        "additionalProperties": False,
    }
    base = _base_variant(search_schema=schema)

    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(("query",)),),
        seed=7,
    )

    source_properties = _properties(_tool(base, "search"))
    assert _tool(transform.variant, "search").input_schema == {
        "type": "object",
        "properties": {
            " ": source_properties[" "],
            "request": {
                "type": "object",
                "properties": {"query": source_properties["query"]},
                "required": ("query",),
                "additionalProperties": False,
            },
        },
        "required": (" ", "request"),
        "additionalProperties": False,
    }


def test_property_named_default_is_not_mistaken_for_a_schema_keyword() -> None:
    schema: dict[str, JSONValue] = {
        "type": "object",
        "properties": {
            "default": {"type": "string"},
            "other": {"type": "integer"},
        },
        "additionalProperties": False,
    }

    transform = build_parameter_restructure_transform(
        _base_variant(search_schema=schema),
        rules=(_search_rule(("default",)),),
        seed=7,
    )

    assert _tool(transform.variant, "search").input_schema["required"] == ("request",)


def test_forbidden_words_in_property_names_and_instance_data_are_allowed() -> None:
    schema: dict[str, JSONValue] = {
        "type": "object",
        "properties": {
            "$ref": {"type": "string"},
            "payload": {
                "const": {
                    "$schema": "PRIVATE_DIALECT",
                    "$id": "literal",
                    "default": "literal",
                    "examples": (),
                },
            },
            "choice": {
                "enum": (
                    {
                        "$schema": "PRIVATE_DIALECT",
                        "$ref": "PRIVATE_REF",
                    },
                ),
            },
        },
        "additionalProperties": False,
    }

    transform = build_parameter_restructure_transform(
        _base_variant(search_schema=schema),
        rules=(_search_rule(("$ref", "choice", "payload")),),
        seed=7,
    )

    target_properties = _properties(_tool(transform.variant, "search"))
    request = target_properties["request"]
    assert isinstance(request, Mapping)
    moved = request["properties"]
    assert isinstance(moved, Mapping)
    assert moved["payload"] == schema["properties"]["payload"]  # type: ignore[index]
    assert moved["choice"] == schema["properties"]["choice"]  # type: ignore[index]


def test_forbidden_keyword_inside_applicator_schema_is_rejected() -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties["query"] = {
        "allOf": (
            {"type": "string"},
            {"not": {"$ref": "#private"}},
        ),
    }

    with pytest.raises(TransformValidationError, match="schema"):
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )


def test_forbidden_keyword_inside_schema_valued_array_is_rejected() -> None:
    schema = _search_schema()
    properties = cast(dict[str, JSONValue], schema["properties"])
    properties["query"] = {
        "type": "array",
        "items": ({"type": "string"}, {"$ref": "#private"}),
    }

    with pytest.raises(TransformValidationError, match="schema"):
        build_parameter_restructure_transform(
            _base_variant(search_schema=schema),
            rules=(_search_rule(("query",)),),
            seed=7,
        )


def test_manifest_paths_preserve_literal_segments() -> None:
    schema: dict[str, JSONValue] = {
        "type": "object",
        "properties": {
            "a.b": {"type": "string"},
            "a/b~c": {"type": "string"},
            "名字": {"type": "string"},
        },
        "additionalProperties": False,
    }
    transform = build_parameter_restructure_transform(
        _base_variant(search_schema=schema),
        rules=(_search_rule(("名字", "a/b~c", "a.b")),),
        seed=7,
    )
    rule_manifest = transform.operator.parameters["rules"]
    assert isinstance(rule_manifest, tuple)
    first = rule_manifest[0]
    assert isinstance(first, Mapping)
    assert first["moves"] == (
        {"source_path": ("a.b",), "surface_path": ("request", "a.b")},
        {"source_path": ("a/b~c",), "surface_path": ("request", "a/b~c")},
        {"source_path": ("名字",), "surface_path": ("request", "名字")},
    )


def test_builder_rejects_post_construction_rule_tamper() -> None:
    rule = _search_rule(("query",))
    object.__setattr__(rule, "container_name", "changed")

    with pytest.raises(TransformValidationError, match="integrity"):
        build_parameter_restructure_transform(_base_variant(), rules=(rule,), seed=7)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("tool_name", "summarize"),
        ("container_name", "changed"),
        ("moved_parameters", ("locale",)),
        ("_canonical_snapshot", b"tampered"),
    ],
)
def test_builder_rejects_every_rule_snapshot_tamper(field_name: str, value: object) -> None:
    rule = _search_rule(("query",))
    object.__setattr__(rule, field_name, value)

    with pytest.raises(TransformValidationError, match="integrity"):
        build_parameter_restructure_transform(_base_variant(), rules=(rule,), seed=7)


def test_direct_transform_rebuilds_and_normalizes_expected_variant() -> None:
    built = build_parameter_restructure_transform(
        _base_variant(),
        rules=(
            ParameterGroupRule("summarize", "payload", ("style",)),
            _search_rule(("query",)),
        ),
        seed=7,
    )

    rebuilt = ParameterRestructureTransform(
        built.source_variant,
        built.variant,
        built.operator,
        tuple(reversed(built.rules)),
    )

    assert rebuilt.rules == built.rules
    assert rebuilt.variant is built.variant
    with pytest.raises(FrozenInstanceError):
        rebuilt.variant = built.source_variant  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(rebuilt)


def test_direct_transform_rejects_operator_parameter_mismatch() -> None:
    built = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(("query",)),),
        seed=7,
    )
    invalid_operator = OperatorManifestEntry(
        built.operator.operator_id,
        built.operator.operator,
        built.operator.level,
        built.operator.seed,
        built.operator.version_hash,
        {"mode": "nest_top_level", "rules": ()},
    )

    with pytest.raises(TransformValidationError, match="parameters"):
        ParameterRestructureTransform(
            built.source_variant,
            built.variant,
            invalid_operator,
            built.rules,
        )


def test_direct_transform_rejects_rule_operator_mismatch() -> None:
    built = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(("query",)),),
        seed=7,
    )

    with pytest.raises(TransformValidationError, match="parameters"):
        ParameterRestructureTransform(
            built.source_variant,
            built.variant,
            built.operator,
            (_search_rule(("limit",)),),
        )


def test_direct_transform_rejects_operator_metadata_mismatch() -> None:
    built = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(("query",)),),
        seed=7,
    )
    invalid_operator = OperatorManifestEntry(
        built.operator.operator_id,
        "other_operator",
        built.operator.level,
        built.operator.seed,
        built.operator.version_hash,
        built.operator.parameters,
    )

    with pytest.raises(TransformValidationError, match="metadata"):
        ParameterRestructureTransform(
            built.source_variant,
            built.variant,
            invalid_operator,
            built.rules,
        )


def test_direct_transform_rejects_variant_not_rebuilt_from_rules() -> None:
    built = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(("query",)),),
        seed=7,
    )
    wrong_variant = SchemaVariant(
        built.variant.variant_id,
        built.source_variant.tools,
        built.variant.manifest,
    )

    with pytest.raises(TransformValidationError, match="variant"):
        ParameterRestructureTransform(
            built.source_variant,
            wrong_variant,
            built.operator,
            built.rules,
        )


def test_direct_transform_rejects_equal_identity_tool_clone() -> None:
    built = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(("query",)),),
        seed=7,
    )
    source_identity = _tool(built.source_variant, "status")
    identity_clone = SurfaceToolSpec(
        source_identity.name,
        source_identity.description,
        source_identity.input_schema,
    )
    cloned_tools = tuple(
        identity_clone if tool.name == source_identity.name else tool
        for tool in built.variant.tools
    )
    equal_variant = SchemaVariant(
        built.variant.variant_id,
        cloned_tools,
        built.variant.manifest,
    )
    assert equal_variant == built.variant
    assert identity_clone is not source_identity

    with pytest.raises(TransformValidationError, match="variant"):
        ParameterRestructureTransform(
            built.source_variant,
            equal_variant,
            built.operator,
            built.rules,
        )


def test_builder_rejects_implicit_composition_with_fixed_error() -> None:
    composed = _base_variant(
        manifest={"kind": "toolshift_interface_variant", "seed": 3},
    )

    with pytest.raises(
        TransformValidationError,
        match=r"^parameter restructure composition requires the explicit compose transform$",
    ):
        build_parameter_restructure_transform(
            composed,
            rules=(_search_rule(("query",)),),
            seed=7,
        )


def test_direct_transform_rejects_implicit_composition_with_fixed_error() -> None:
    clean = _base_variant()
    built = build_parameter_restructure_transform(
        clean,
        rules=(_search_rule(("query",)),),
        seed=7,
    )
    composed = _base_variant(
        manifest={"kind": "toolshift_interface_variant", "seed": 3},
    )
    operator = OperatorManifestEntry(
        "restructure-000",
        "parameter_restructure",
        "L2",
        derive_operator_seed(
            7,
            schema_fingerprint(composed),
            "restructure-000",
            "parameter_restructure",
            0,
        ),
        PARAMETER_RESTRUCTURE_VERSION_HASH,
        built.operator.parameters,
    )
    variant = build_transformed_variant(
        composed,
        built.variant.tools,
        seed=7,
        operators=(operator,),
    )

    with pytest.raises(
        TransformValidationError,
        match=r"^parameter restructure composition requires the explicit compose transform$",
    ):
        ParameterRestructureTransform(composed, variant, operator, built.rules)


def test_variant_id_is_the_full_manifest_digest() -> None:
    base = _base_variant()
    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )

    assert transform.source_variant is base
    assert transform.variant.variant_id == (
        f"toolshift-v1-{manifest_sha256(transform.variant.manifest)}"
    )
    assert set(transform.variant.manifest) == {
        "schema_version",
        "kind",
        "base_schema_fingerprint",
        "seed",
        "composition_order",
        "version_hash",
        "operators",
    }
    assert transform.variant.manifest["composition_order"] == ("restructure-000",)


@pytest.mark.parametrize(
    "explicit_value",
    [
        None,
        False,
        0,
        "",
        [],
        {},
        ["值", {"nested": [0, False, None]}],
    ],
)
def test_call_translation_round_trip_preserves_falsey_values_and_metadata(
    explicit_value: JSONValue,
) -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )
    canonical_call: dict[str, JSONValue] = {
        "name": "search",
        "arguments": {
            "query": "雪",
            "limit": explicit_value,
            "locale": "zh-CN",
        },
        "request_id": "请求-1",
        "metadata": {"attempt": 0, "flags": [False, None]},
    }
    original = copy.deepcopy(canonical_call)

    surface = transform.canonical_call_to_surface(canonical_call)
    surface_arguments = surface["arguments"]
    assert isinstance(surface_arguments, Mapping)
    request = surface_arguments["request"]
    assert isinstance(request, Mapping)
    assert request["query"] == "雪"
    assert "limit" in request
    assert canonical_json_bytes(request["limit"]) == canonical_json_bytes(explicit_value)
    assert surface_arguments["locale"] == "zh-CN"
    assert surface["request_id"] == "请求-1"
    assert canonical_json_bytes(surface["metadata"]) == canonical_json_bytes(
        canonical_call["metadata"]
    )

    round_trip = transform.surface_call_to_canonical(surface)
    assert canonical_json_bytes(round_trip) == canonical_json_bytes(canonical_call)
    assert canonical_call == original
    with pytest.raises(TypeError):
        surface_arguments["changed"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        request["changed"] = True  # type: ignore[index]


def test_call_translation_preserves_optional_absence_and_creates_empty_container() -> None:
    base = _base_variant()
    selective = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )
    without_limit = {
        "name": "search",
        "arguments": {"query": "alpha", "locale": None},
    }

    surface = selective.canonical_call_to_surface(without_limit)
    surface_arguments = surface["arguments"]
    assert isinstance(surface_arguments, Mapping)
    request = surface_arguments["request"]
    assert isinstance(request, Mapping)
    assert set(request) == {"query"}
    assert "limit" not in request
    assert canonical_json_bytes(selective.surface_call_to_canonical(surface)) == (
        canonical_json_bytes(without_limit)
    )

    optional_only = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(("limit",)),),
        seed=7,
    )
    optional_surface = optional_only.canonical_call_to_surface(without_limit)
    optional_arguments = optional_surface["arguments"]
    assert isinstance(optional_arguments, Mapping)
    assert optional_arguments["request"] == {}
    assert canonical_json_bytes(optional_only.surface_call_to_canonical(optional_surface)) == (
        canonical_json_bytes(without_limit)
    )


def test_whole_wrap_call_translation_moves_every_present_parameter() -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(("query", "limit", "locale")),),
        seed=7,
    )
    canonical = {
        "name": "search",
        "arguments": {"query": "alpha", "locale": None},
    }

    surface = transform.canonical_call_to_surface(canonical)
    assert surface["arguments"] == {
        "request": {"query": "alpha", "locale": None},
    }
    assert canonical_json_bytes(transform.surface_call_to_canonical(surface)) == (
        canonical_json_bytes(canonical)
    )


def test_identity_tool_calls_are_frozen_without_parameter_mode_classification() -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )
    identity_call = {
        "name": "status",
        "arguments": {
            "request": {"query": "literal"},
            "query": "also-literal",
            "nested": [1, {"ok": False}],
        },
        "metadata": {"empty": {}},
    }

    canonical = transform.surface_call_to_canonical(identity_call)
    surface = transform.canonical_call_to_surface(identity_call)

    assert canonical_json_bytes(canonical) == canonical_json_bytes(identity_call)
    assert canonical_json_bytes(surface) == canonical_json_bytes(identity_call)
    assert canonical is not identity_call
    assert surface is not identity_call
    arguments = canonical["arguments"]
    assert isinstance(arguments, Mapping)
    with pytest.raises(TypeError):
        arguments["changed"] = True  # type: ignore[index]


@pytest.mark.parametrize(
    "identity_call",
    [
        {"name": "status", "arguments": "ping", "metadata": {"attempt": 0}},
        {"name": "status", "arguments": None, "metadata": {"attempt": 0}},
        {"name": "status", "metadata": {"attempt": 0}},
    ],
)
def test_identity_tool_preserves_scalar_null_and_missing_argument_shapes(
    identity_call: dict[str, JSONValue],
) -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )
    original = copy.deepcopy(identity_call)

    canonical = transform.surface_call_to_canonical(identity_call)
    surface = transform.canonical_call_to_surface(identity_call)

    assert canonical_json_bytes(canonical) == canonical_json_bytes(identity_call)
    assert canonical_json_bytes(surface) == canonical_json_bytes(identity_call)
    assert canonical is not identity_call
    assert surface is not identity_call
    assert identity_call == original


@pytest.mark.parametrize(
    "identity_call",
    [
        {"name": "status", "arguments": "ping"},
        {"name": "status"},
    ],
)
def test_identity_argument_shapes_delegate_parse_and_wrap_exactly_once(
    identity_call: dict[str, JSONValue],
) -> None:
    source = _IdentityArgumentShapeSource(_base_variant())
    adapter = _apply_restructure(source)

    actions = adapter.surface_to_semantic(identity_call)
    assert actions is source.parse_results[0]
    assert len(source.parse_inputs) == 1
    assert canonical_json_bytes(source.parse_inputs[0]) == canonical_json_bytes(identity_call)

    observation: JSONValue = {"status": "ok"}
    groups = ((observation,),)
    wrapped = adapter.base_observation_to_surface(identity_call, actions, groups)
    delegated_call, delegated_actions, delegated_groups = source.wrap_inputs[0]
    assert canonical_json_bytes(delegated_call) == canonical_json_bytes(identity_call)
    assert delegated_actions is actions
    assert delegated_groups is groups
    assert wrapped is observation
    assert len(source.wrap_inputs) == 1


@pytest.mark.parametrize(
    "identity_call",
    [
        {"name": "status", "arguments": "ping"},
        {"name": "status"},
    ],
)
def test_identity_argument_shapes_keep_neutral_trace_identity(
    identity_call: dict[str, JSONValue],
) -> None:
    source = _IdentityArgumentShapeSource(_base_variant())
    adapter = _apply_restructure(source)
    trace = ExecutionTrace((identity_call,), (), ())

    result = adapter.canonicalize_trace(trace)

    assert result is trace.semantic_actions
    assert len(source.trace_inputs) == 1
    assert source.trace_inputs[0] is trace


def test_identity_argument_shapes_still_reject_invalid_i_json() -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )
    invalid_call = {"name": "status", "arguments": float("nan")}

    for translate in (
        transform.surface_call_to_canonical,
        transform.canonical_call_to_surface,
    ):
        with pytest.raises(TransformValidationError) as caught:
            translate(invalid_call)
        assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "call",
    [
        {"name": "unknown", "arguments": {}},
        {"arguments": {"locale": "en", "request": {"query": "alpha"}}},
        {"name": 7, "arguments": {}},
        {"name": "search"},
        {"name": "search", "arguments": "PRIVATE_ARGUMENTS"},
        {"name": "search", "arguments": {"query": "alpha", "locale": "en"}},
        {
            "name": "search",
            "arguments": {"locale": "en", "request": "PRIVATE_CONTAINER"},
        },
        {
            "name": "search",
            "arguments": {
                "query": "hybrid",
                "locale": "en",
                "request": {"query": "alpha"},
            },
        },
        {
            "name": "search",
            "arguments": {
                "locale": "en",
                "PRIVATE_ROOT": True,
                "request": {"query": "alpha"},
            },
        },
        {
            "name": "search",
            "arguments": {
                "locale": "en",
                "request": {"query": "alpha", "PRIVATE_INNER": True},
            },
        },
        {"name": "search", "arguments": {"request": {"query": "alpha"}}},
        {"name": "search", "arguments": {"locale": "en", "request": {}}},
    ],
)
def test_surface_call_translation_rejects_invalid_closed_shapes_without_payload(
    call: object,
) -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )

    with pytest.raises(TransformValidationError) as caught:
        transform.surface_call_to_canonical(call)  # type: ignore[arg-type]

    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "call",
    [
        {"name": "unknown", "arguments": {}},
        {"arguments": {"query": "alpha", "locale": "en"}},
        {"name": False, "arguments": {}},
        {"name": "search"},
        {"name": "search", "arguments": "PRIVATE_ARGUMENTS"},
        {
            "name": "search",
            "arguments": {"query": "alpha", "locale": "en", "request": {}},
        },
        {
            "name": "search",
            "arguments": {"query": "alpha", "locale": "en", "PRIVATE_ROOT": True},
        },
        {"name": "search", "arguments": {"locale": "en"}},
        {"name": "search", "arguments": {"query": "alpha"}},
        {"name": "search", "arguments": {"query": "alpha", "locale": float("nan")}},
        _ExplodingCallMapping(name="search", arguments={}),
    ],
)
def test_canonical_call_translation_rejects_invalid_calls_without_payload(
    call: object,
) -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )

    with pytest.raises(TransformValidationError) as caught:
        transform.canonical_call_to_surface(call)  # type: ignore[arg-type]

    assert "PRIVATE" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_trace_modes_preserve_canonical_and_neutral_identity_and_translate_surface() -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(
            _search_rule(),
            ParameterGroupRule("summarize", "payload", ("text",)),
        ),
        seed=7,
    )
    search_action = SemanticAction("search", {"query": "alpha", "locale": "en"})
    canonical_search = {
        "name": "search",
        "arguments": {"query": "alpha", "locale": "en"},
    }
    surface_search = {
        "name": "search",
        "arguments": {"locale": "en", "request": {"query": "alpha"}},
    }
    identity_call = {
        "name": "status",
        "arguments": {"request": {"query": "literal"}, "query": "literal"},
    }
    base_call = {"name": "search", "arguments": {"query": "alpha", "locale": "en"}}
    canonical_trace = ExecutionTrace(
        (canonical_search, identity_call),
        (search_action,),
        (base_call,),
    )
    neutral_trace = ExecutionTrace((identity_call,), (), ())
    surface_trace = ExecutionTrace(
        (surface_search, identity_call),
        (search_action,),
        (base_call,),
    )

    assert transform._trace_for_source(canonical_trace) is canonical_trace
    assert transform._trace_for_source(neutral_trace) is neutral_trace
    translated = transform._trace_for_source(surface_trace)
    assert translated is not surface_trace
    assert canonical_json_bytes(translated.surface_calls[0]) == canonical_json_bytes(
        canonical_search
    )
    assert canonical_json_bytes(translated.surface_calls[1]) == canonical_json_bytes(identity_call)
    assert translated.semantic_actions is surface_trace.semantic_actions
    assert translated.semantic_actions[0] is search_action
    assert canonical_json_bytes(translated.base_calls) == canonical_json_bytes(
        surface_trace.base_calls
    )


@pytest.mark.parametrize(
    "surface_calls",
    [
        (
            {
                "name": "search",
                "arguments": {"locale": "en", "request": {"query": "alpha"}},
            },
            {"name": "summarize", "arguments": {"text": "alpha"}},
        ),
        (
            {
                "name": "search",
                "arguments": {
                    "query": "hybrid",
                    "locale": "en",
                    "request": {"query": "alpha"},
                },
            },
        ),
        ({"name": "unknown", "arguments": {}},),
        ({"name": "search", "arguments": {"query": "alpha"}},),
        (
            {
                "name": "search",
                "arguments": {
                    "locale": "en",
                    "request": {"query": "alpha", "PRIVATE_INNER": True},
                },
            },
        ),
    ],
)
def test_trace_rejects_mixed_hybrid_unknown_and_malformed_calls_before_source(
    surface_calls: tuple[Mapping[str, JSONValue], ...],
) -> None:
    source = _RecordingRestructureSource(_base_variant())
    adapter = _apply_restructure(
        source,
        rules=(
            _search_rule(),
            ParameterGroupRule("summarize", "payload", ("text",)),
        ),
    )
    trace = ExecutionTrace(surface_calls, (), ())

    with pytest.raises(TransformValidationError) as caught:
        adapter.canonicalize_trace(trace)

    assert "PRIVATE" not in str(caught.value)
    assert source.trace_inputs == []


class _StageErrorRestructureSource(_RecordingRestructureSource):
    def __init__(self, variant: SchemaVariant, stage: str, error_type: type[Exception]) -> None:
        super().__init__(variant)
        self._stage = stage
        self.error = error_type("PRIVATE_STAGE_PAYLOAD")

    def _maybe_raise(self, stage: str) -> None:
        if self._stage == stage:
            raise self.error

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        self._maybe_raise("parse")
        return super().surface_to_semantic(surface_call)

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self._maybe_raise("compile")
        return super().semantic_to_base_calls(action)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self._maybe_raise("wrap")
        return super().base_observation_to_surface(
            surface_call,
            actions,
            base_observation_groups,
        )

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self._maybe_raise("canonicalize")
        return super().canonicalize_trace(trace)


class _RebindingRestructureSource(_RecordingRestructureSource):
    def __init__(self, variant: SchemaVariant, stage: str) -> None:
        super().__init__(variant)
        self._stage = stage
        self._replacement = SchemaVariant(
            "replacement-v1",
            variant.tools,
            {"operator": "replacement", "seed": 9},
        )

    def _rebind(self, stage: str) -> None:
        if self._stage == stage:
            self._variant = self._replacement

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        result = super().surface_to_semantic(surface_call)
        self._rebind("parse")
        return result

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        result = super().semantic_to_base_calls(action)
        self._rebind("compile")
        return result

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        result = super().base_observation_to_surface(
            surface_call,
            actions,
            base_observation_groups,
        )
        self._rebind("wrap")
        return result

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        result = super().canonicalize_trace(trace)
        self._rebind("canonicalize")
        return result


class _RebindThenRaiseRestructureSource(_RebindingRestructureSource):
    def __init__(self, variant: SchemaVariant, stage: str, error_type: type[Exception]) -> None:
        super().__init__(variant, stage)
        self.error = error_type("PRIVATE_REBIND_PAYLOAD")

    def _raise_after_rebind(self, stage: str) -> None:
        self._rebind(stage)
        raise self.error

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        self._raise_after_rebind("parse")

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self._raise_after_rebind("compile")

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self._raise_after_rebind("wrap")

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self._raise_after_rebind("canonicalize")


class _PublicLieRestructureSource(_RecordingRestructureSource):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._public_variant = _base_variant()

    @property
    def variant(self) -> SchemaVariant:
        return self._public_variant


class _PayloadPublicRestructureSource(_RecordingRestructureSource):
    def __init__(self, variant: SchemaVariant) -> None:
        self.raise_public = False
        super().__init__(variant)

    @property
    def variant(self) -> SchemaVariant:
        if self.raise_public:
            raise RuntimeError("PRIVATE_PUBLIC_VARIANT_PAYLOAD")
        return super().variant


def _invoke_restructure_stage(adapter: object, stage: str) -> object:
    surface_call = {
        "name": "search",
        "arguments": {"locale": "en", "request": {"query": "alpha"}},
    }
    canonical_call = {
        "name": "search",
        "arguments": {"query": "alpha", "locale": "en"},
    }
    action = SemanticAction("search", {"query": "alpha", "locale": "en"})
    if stage == "parse":
        return adapter.surface_to_semantic(surface_call)  # type: ignore[attr-defined]
    if stage == "compile":
        return adapter.semantic_to_base_calls(action)  # type: ignore[attr-defined]
    if stage == "wrap":
        return adapter.base_observation_to_surface(  # type: ignore[attr-defined]
            surface_call,
            (action,),
            (({"ok": True},),),
        )
    return adapter.canonicalize_trace(  # type: ignore[attr-defined]
        ExecutionTrace((surface_call,), (action,), (canonical_call,))
    )


def test_parameter_restructure_adapter_delegates_pipeline_objects_exactly_once() -> None:
    source = _RecordingRestructureSource(_base_variant())
    adapter = _apply_restructure(source)
    surface_call = {
        "name": "search",
        "arguments": {
            "locale": "zh-CN",
            "request": {"query": "雪", "limit": 0},
        },
        "metadata": {"retry": False},
    }

    actions = adapter.surface_to_semantic(surface_call)
    assert actions is source.parse_results[0]
    delegated_parse = source.parse_inputs[0]
    assert delegated_parse == {
        "name": "search",
        "arguments": {"locale": "zh-CN", "query": "雪", "limit": 0},
        "metadata": {"retry": False},
    }

    base_calls = adapter.semantic_to_base_calls(actions[0])
    assert source.compile_inputs == [actions[0]]
    assert source.compile_inputs[0] is actions[0]
    assert base_calls is source.compile_results[0]

    observation: JSONValue = {"items": ["雪", False, None]}
    groups = ((observation,),)
    wrapped = adapter.base_observation_to_surface(surface_call, actions, groups)
    delegated_call, delegated_actions, delegated_groups = source.wrap_inputs[0]
    assert delegated_call == delegated_parse
    assert delegated_actions is actions
    assert delegated_groups is groups
    assert wrapped is observation
    assert len(source.parse_inputs) == len(source.compile_inputs) == len(source.wrap_inputs) == 1


def test_adapter_canonicalize_trace_preserves_source_trace_and_action_identity() -> None:
    source = _RecordingRestructureSource(_base_variant())
    adapter = _apply_restructure(source)
    action = SemanticAction("search", {"query": "alpha", "locale": "en"})
    canonical_call = {
        "name": "search",
        "arguments": {"query": "alpha", "locale": "en"},
    }
    surface_call = {
        "name": "search",
        "arguments": {"locale": "en", "request": {"query": "alpha"}},
    }
    canonical_trace = ExecutionTrace((canonical_call,), (action,), (canonical_call,))
    neutral_trace = ExecutionTrace(
        ({"name": "status", "arguments": {"request": {}, "query": "literal"}},),
        (),
        (),
    )
    surface_trace = ExecutionTrace((surface_call,), (action,), (canonical_call,))

    canonical_result = adapter.canonicalize_trace(canonical_trace)
    neutral_result = adapter.canonicalize_trace(neutral_trace)
    surface_result = adapter.canonicalize_trace(surface_trace)

    assert canonical_result is canonical_trace.semantic_actions
    assert neutral_result is neutral_trace.semantic_actions
    assert surface_result is surface_trace.semantic_actions
    assert source.trace_inputs[0] is canonical_trace
    assert source.trace_inputs[1] is neutral_trace
    delegated_surface = source.trace_inputs[2]
    assert delegated_surface is not surface_trace
    assert delegated_surface.semantic_actions is surface_trace.semantic_actions
    assert delegated_surface.semantic_actions[0] is action
    assert canonical_json_bytes(delegated_surface.surface_calls[0]) == canonical_json_bytes(
        canonical_call
    )
    assert canonical_json_bytes(delegated_surface.base_calls) == canonical_json_bytes(
        surface_trace.base_calls
    )


@pytest.mark.parametrize("stage", ["parse", "compile", "wrap", "canonicalize"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_source_errors_follow_stage_specific_static_delegation_policy(
    stage: str,
    error_type: type[Exception],
) -> None:
    source = _StageErrorRestructureSource(_base_variant(), stage, error_type)
    adapter = _apply_restructure(source)
    expected_errors = {
        "parse": "source adapter rejected the translated surface call",
        "compile": "source adapter rejected the semantic action",
        "wrap": "source adapter rejected observation wrapping",
        "canonicalize": "source adapter rejected trace canonicalization",
    }

    if error_type is ValueError:
        with pytest.raises(TransformValidationError) as caught:
            _invoke_restructure_stage(adapter, stage)
        assert str(caught.value) == expected_errors[stage]
        assert caught.value.__cause__ is None
    else:
        with pytest.raises(RuntimeError) as caught:
            _invoke_restructure_stage(adapter, stage)
        assert caught.value is source.error


@pytest.mark.parametrize("stage", ["parse", "compile", "wrap", "canonicalize"])
@pytest.mark.parametrize("outcome", ["return", "value_error", "runtime_error"])
def test_final_binding_failure_overrides_every_stage_callback_outcome(
    stage: str,
    outcome: str,
) -> None:
    if outcome == "return":
        source: _RecordingRestructureSource = _RebindingRestructureSource(
            _base_variant(),
            stage,
        )
    else:
        error_type = ValueError if outcome == "value_error" else RuntimeError
        source = _RebindThenRaiseRestructureSource(_base_variant(), stage, error_type)
    adapter = _apply_restructure(source)

    with pytest.raises(TransformValidationError, match="binding") as caught:
        _invoke_restructure_stage(adapter, stage)

    assert "PRIVATE" not in str(caught.value)


def test_preexisting_binding_failure_prevents_source_callback() -> None:
    base = _base_variant()
    source = _RecordingRestructureSource(base)
    adapter = _apply_restructure(source)
    object.__setattr__(source, "_variant", _base_variant())

    with pytest.raises(TransformValidationError, match="binding"):
        adapter.surface_to_semantic(
            {
                "name": "search",
                "arguments": {"locale": "en", "request": {"query": "alpha"}},
            }
        )

    assert source.parse_inputs == []


def test_adapter_construction_requires_exact_raw_and_public_source_binding() -> None:
    base = _base_variant()
    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )
    equal_but_distinct = _base_variant()

    with pytest.raises(TransformValidationError, match="binding"):
        restructure_module.ParameterRestructureAdapter(
            _RecordingRestructureSource(equal_but_distinct),
            transform,
        )
    with pytest.raises(TransformValidationError, match="binding"):
        restructure_module.ParameterRestructureAdapter(
            _PublicLieRestructureSource(base),
            transform,
        )


def test_public_variant_callback_payload_is_sanitized_at_init_and_runtime() -> None:
    base = _base_variant()
    initial_source = _PayloadPublicRestructureSource(base)
    initial_source.raise_public = True
    transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )
    with pytest.raises(TransformValidationError) as initial_error:
        restructure_module.ParameterRestructureAdapter(initial_source, transform)
    assert "PRIVATE_PUBLIC_VARIANT_PAYLOAD" not in str(initial_error.value)

    runtime_source = _PayloadPublicRestructureSource(base)
    adapter = restructure_module.ParameterRestructureAdapter(runtime_source, transform)
    runtime_source.raise_public = True
    with pytest.raises(TransformValidationError) as runtime_error:
        adapter.surface_to_semantic(
            {
                "name": "search",
                "arguments": {"locale": "en", "request": {"query": "alpha"}},
            }
        )
    assert "PRIVATE_PUBLIC_VARIANT_PAYLOAD" not in str(runtime_error.value)


def test_runtime_binding_rejects_adapter_transform_and_final_variant_replacement() -> None:
    base = _base_variant()
    source = _RecordingRestructureSource(base)
    adapter = _apply_restructure(source)
    valid_call = {
        "name": "search",
        "arguments": {"locale": "en", "request": {"query": "alpha"}},
    }

    object.__setattr__(adapter, "_source_adapter", _RecordingRestructureSource(base))
    with pytest.raises(TransformValidationError, match="binding"):
        adapter.surface_to_semantic(valid_call)

    source = _RecordingRestructureSource(base)
    adapter = _apply_restructure(source)
    replacement_transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )
    object.__setattr__(adapter, "_transform", replacement_transform)
    with pytest.raises(TransformValidationError, match="binding"):
        adapter.surface_to_semantic(valid_call)

    source = _RecordingRestructureSource(base)
    adapter = _apply_restructure(source)
    object.__setattr__(adapter, "_variant", base)
    with pytest.raises(TransformValidationError, match="binding"):
        adapter.surface_to_semantic(valid_call)


def test_transform_runtime_seals_reject_root_replacement() -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )
    object.__setattr__(transform, "source_variant", _base_variant())

    with pytest.raises(TransformValidationError, match="integrity"):
        transform.canonical_call_to_surface(
            {
                "name": "search",
                "arguments": {"query": "alpha", "locale": "en"},
            }
        )


def test_transform_runtime_lookup_index_tamper_fails_closed_for_changed_tool() -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )
    lookup = object.__getattribute__(transform, "_plans_by_tool")
    index = object.__getattribute__(lookup, "_index")
    index.clear()

    with pytest.raises(TransformValidationError, match="integrity"):
        transform.canonical_call_to_surface(
            {
                "name": "search",
                "arguments": {"query": "alpha", "locale": "en"},
            }
        )


def test_helper_and_direct_adapter_reject_implicit_composition() -> None:
    first = _apply_restructure()

    with pytest.raises(
        TransformValidationError,
        match=r"^parameter restructure composition requires the explicit compose transform$",
    ):
        restructure_module.apply_parameter_restructure(
            first,
            rules=(_search_rule(),),
            seed=8,
        )

    clean_base = _base_variant()
    clean_transform = build_parameter_restructure_transform(
        clean_base,
        rules=(_search_rule(),),
        seed=8,
    )
    with pytest.raises(
        TransformValidationError,
        match=r"^parameter restructure composition requires the explicit compose transform$",
    ):
        restructure_module.ParameterRestructureAdapter(first, clean_transform)


def test_rename_to_restructure_composition_is_classified_at_every_entry() -> None:
    base = _base_variant()
    rename_adapter = rename_module.apply_rename(
        _RecordingRestructureSource(base),
        tool_name_mapping={"search": "lookup"},
        seed=13,
    )
    clean_transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )
    composed_rule = ParameterGroupRule("lookup", "request", ("query", "limit"))
    expected = r"^parameter restructure composition requires the explicit compose transform$"

    with pytest.raises(TransformValidationError, match=expected):
        build_parameter_restructure_transform(
            rename_adapter.variant,
            rules=(composed_rule,),
            seed=7,
        )
    with pytest.raises(TransformValidationError, match=expected):
        ParameterRestructureTransform(
            rename_adapter.variant,
            clean_transform.variant,
            clean_transform.operator,
            clean_transform.rules,
        )
    with pytest.raises(TransformValidationError, match=expected):
        restructure_module.apply_parameter_restructure(
            rename_adapter,
            rules=(composed_rule,),
            seed=7,
        )
    with pytest.raises(TransformValidationError, match=expected):
        restructure_module.ParameterRestructureAdapter(rename_adapter, clean_transform)


def test_restructure_to_rename_composition_is_classified_at_every_entry() -> None:
    base = _base_variant()
    restructure_adapter = _apply_restructure(_RecordingRestructureSource(base))
    clean_transform = rename_module.build_rename_transform(
        base,
        tool_name_mapping={"search": "lookup"},
        seed=13,
    )
    expected = r"^rename composition requires the explicit compose transform$"

    with pytest.raises(TransformValidationError, match=expected):
        rename_module.build_rename_transform(
            restructure_adapter.variant,
            tool_name_mapping={"search": "lookup"},
            seed=13,
        )
    with pytest.raises(TransformValidationError, match=expected):
        rename_module.RenameTransform(
            restructure_adapter.variant,
            clean_transform.variant,
            clean_transform.operator,
            clean_transform.canonical_to_surface,
        )
    with pytest.raises(TransformValidationError, match=expected):
        rename_module.apply_rename(
            restructure_adapter,
            tool_name_mapping={"search": "lookup"},
            seed=13,
        )
    with pytest.raises(TransformValidationError, match=expected):
        rename_module.RenameAdapter(restructure_adapter, clean_transform)


def test_cross_composition_manifest_probes_sanitize_public_callback_payloads() -> None:
    base = _base_variant()
    source = _ManifestPayloadPublicSource(base)
    restructure_transform = build_parameter_restructure_transform(
        base,
        rules=(_search_rule(),),
        seed=7,
    )
    rename_transform = rename_module.build_rename_transform(
        base,
        tool_name_mapping={"search": "lookup"},
        seed=13,
    )

    for constructor, transform in (
        (restructure_module.ParameterRestructureAdapter, restructure_transform),
        (rename_module.RenameAdapter, rename_transform),
    ):
        with pytest.raises(TransformValidationError) as caught:
            constructor(source, transform)
        assert "PRIVATE_MANIFEST_PAYLOAD" not in str(caught.value)
        assert caught.value.__cause__ is None


def test_cross_composition_manifest_probes_sanitize_raw_callback_payloads() -> None:
    raw_variant = _base_variant()
    object.__setattr__(raw_variant, "manifest", _ExplodingManifestMapping())
    source = _RecordingRestructureSource(raw_variant)

    for apply in (
        restructure_module.apply_parameter_restructure,
        rename_module.apply_rename,
    ):
        kwargs: dict[str, object]
        if apply is restructure_module.apply_parameter_restructure:
            kwargs = {"rules": (_search_rule(),), "seed": 7}
        else:
            kwargs = {"tool_name_mapping": {"search": "lookup"}, "seed": 13}
        with pytest.raises(TransformValidationError) as caught:
            apply(source, **kwargs)
        assert "PRIVATE_MANIFEST_PAYLOAD" not in str(caught.value)
        assert caught.value.__cause__ is None


def test_online_translation_uses_only_precomputed_plans_and_runtime_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = build_parameter_restructure_transform(
        _base_variant(),
        rules=(_search_rule(),),
        seed=7,
    )

    def unexpected_construction_read(*args: object, **kwargs: object) -> object:
        pytest.fail("online translation performed construction-time schema work")

    monkeypatch.setattr(restructure_module, "schema_fingerprint", unexpected_construction_read)
    monkeypatch.setattr(restructure_module, "canonical_json_bytes", unexpected_construction_read)
    monkeypatch.setattr(restructure_module, "_admit_changed_schema", unexpected_construction_read)
    monkeypatch.setattr(
        restructure_module,
        "_build_restructured_tools",
        unexpected_construction_read,
    )
    monkeypatch.setattr(
        restructure_module,
        "_build_translation_plans",
        unexpected_construction_read,
    )
    monkeypatch.setattr(
        restructure_module,
        "_make_schema_runtime_seal",
        unexpected_construction_read,
        raising=False,
    )
    monkeypatch.setattr(
        restructure_module,
        "_make_operator_runtime_seal",
        unexpected_construction_read,
        raising=False,
    )
    monkeypatch.setattr(
        restructure_module,
        "_make_mapping_root_seal",
        unexpected_construction_read,
        raising=False,
    )

    canonical = {
        "name": "search",
        "arguments": {"query": "alpha", "locale": "en"},
    }
    for _ in range(5):
        surface = transform.canonical_call_to_surface(canonical)
        assert canonical_json_bytes(transform.surface_call_to_canonical(surface)) == (
            canonical_json_bytes(canonical)
        )
