"""Schema and manifest tests for deterministic L2 parameter grouping."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

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
from toolshift.types import JSONValue, SchemaVariant, SurfaceToolSpec, manifest_sha256


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


def test_parameter_restructure_abi_hash_has_fixed_vector() -> None:
    assert (
        PARAMETER_RESTRUCTURE_VERSION_HASH
        == "6a170c4eb544cda3798a4106d30625e9cf25b46878fbea7e27589895348d45ac"
    )


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
