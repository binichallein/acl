"""Synthetic-only tests for the pure AppWorld semantic adapter."""

from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry

import toolshift.adapters as adapters_module
import toolshift.adapters.appworld as appworld_adapter_module
from toolshift.adapters import AppWorldSemanticAdapter, build_appworld_adapter
from toolshift.types import ExecutionTrace, SemanticAction, canonical_json_bytes


def _tool(
    name: str,
    *,
    description: str = "Synthetic tool.",
    properties: Mapping[str, object] | None = None,
    required: list[str] | None = None,
    additional_properties: object = None,
) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "object",
        "properties": dict(properties or {}),
        "required": list(required or []),
    }
    if additional_properties is not None:
        schema["additionalProperties"] = additional_properties
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


def _catalog() -> list[dict[str, object]]:
    return [
        _tool(
            "mail__send__draft",
            properties={
                "body": {
                    "type": "string",
                    "description": "Synthetic body.",
                    "minLength": 1,
                },
                "priority": {
                    "type": "integer",
                    "description": "Synthetic priority.",
                    "default": None,
                },
            },
            required=["body"],
        ),
        _tool(
            "calendar__list_events",
            properties={"limit": {"type": "integer", "minimum": 1, "default": None}},
            additional_properties=False,
        ),
    ]


def _semantic_catalog() -> list[dict[str, object]]:
    return [
        _tool(
            "calendar__create_event",
            properties={
                "title": {"type": "string", "minLength": 1},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string", "format": "email"},
                },
                "options": {
                    "type": "object",
                    "properties": {"notify": {"type": "boolean"}},
                    "required": ["notify"],
                    "additionalProperties": False,
                },
                "retries": {"type": "integer", "minimum": 0, "default": 2},
            },
            required=["title", "attendees", "options"],
        ),
        _tool(
            "mail__send_message",
            properties={"body": {"type": "string", "minLength": 1}},
            required=["body"],
        ),
    ]


def _event_call(*, metadata: object | None = None) -> dict[str, object]:
    call: dict[str, object] = {
        "name": "calendar__create_event",
        "arguments": {
            "title": "Synthetic meeting",
            "attendees": ["synthetic@example.com"],
            "options": {"notify": True},
        },
    }
    if metadata is not None:
        call["metadata"] = metadata
    return call


def test_build_appworld_adapter_snapshots_sorts_and_closes_catalog() -> None:
    catalog = _catalog()
    original = copy.deepcopy(catalog)

    adapter = build_appworld_adapter(catalog)

    assert isinstance(adapter, AppWorldSemanticAdapter)
    assert adapter.variant.manifest == {
        "kind": "appworld_source_interface",
        "schema_policy": "closed-root-v1",
    }
    assert tuple(tool.name for tool in adapter.variant.tools) == (
        "calendar__list_events",
        "mail__send__draft",
    )
    assert all(tool.input_schema["additionalProperties"] is False for tool in adapter.variant.tools)
    send = adapter.variant.tools[1]
    assert send.description == "Synthetic tool."
    assert send.input_schema["required"] == ("body",)
    assert send.input_schema["properties"]["priority"]["default"] is None
    assert send.input_schema["properties"]["body"]["minLength"] == 1
    assert catalog == original


def test_appworld_adapter_can_be_constructed_directly() -> None:
    direct = AppWorldSemanticAdapter(_catalog())
    built = build_appworld_adapter(_catalog())

    assert direct.variant == built.variant


def test_build_appworld_adapter_delegates_to_the_class_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    constructed: list[object] = []

    class ConstructorSpy:
        def __init__(self, supplied: object) -> None:
            constructed.append(supplied)

    monkeypatch.setattr(appworld_adapter_module, "AppWorldSemanticAdapter", ConstructorSpy)

    adapter = appworld_adapter_module.build_appworld_adapter(catalog)

    assert type(adapter) is ConstructorSpy
    assert constructed == [catalog]


def test_catalog_construction_installs_one_format_checker_per_schema() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "contacts__create_contact",
                properties={"email": {"type": "string", "format": "email"}},
                required=["email"],
            ),
            _tool(
                "calendar__invite_attendee",
                properties={"email": {"type": "string", "format": "email"}},
                required=["email"],
            ),
        ]
    )

    validators = object.__getattribute__(adapter, "_validators")
    assert isinstance(validators, Mapping)
    first, second = tuple(validators.values())
    assert type(first) is Draft202012Validator
    assert type(second) is Draft202012Validator
    assert isinstance(first.format_checker, FormatChecker)
    assert isinstance(second.format_checker, FormatChecker)
    assert first.format_checker is not second.format_checker
    assert first.is_valid({"email": "not-an-email"}) is False
    assert first.is_valid({"email": "synthetic@example.com"}) is True


@pytest.mark.parametrize(
    ("format_name", "valid_value", "invalid_value"),
    [
        (
            "date-time",
            "2000-01-01T00:00:00Z",
            "PRIVATE_INVALID_DATETIME_CANARY",
        ),
        ("uri", "https://example.com/synthetic", "PRIVATE INVALID URI CANARY"),
    ],
)
def test_catalog_construction_enforces_draft202012_formats(
    format_name: str,
    valid_value: str,
    invalid_value: str,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__validate_value",
                properties={"value": {"type": "string", "format": format_name}},
                required=["value"],
            )
        ]
    )

    assert adapter.surface_to_semantic(
        {
            "name": "synthetic__validate_value",
            "arguments": {"value": valid_value},
        }
    ) == (SemanticAction("synthetic__validate_value", {"value": valid_value}),)
    with pytest.raises(ValueError, match=r"^AppWorld surface call is invalid$") as raised:
        adapter.surface_to_semantic(
            {
                "name": "synthetic__validate_value",
                "arguments": {"value": invalid_value},
            }
        )

    assert invalid_value not in str(raised.value)


def test_catalog_construction_rejects_unknown_format_without_payload() -> None:
    private_format = "PRIVATE_UNKNOWN_FORMAT_CANARY"

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter(
            [
                _tool(
                    "synthetic__validate_value",
                    properties={
                        "value": {"type": "string", "format": private_format}
                    },
                )
            ]
        )

    assert private_format not in str(raised.value)


@pytest.mark.parametrize(
    "property_schema",
    [
        {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "string",
                    "format": "PRIVATE_NESTED_FORMAT_CANARY",
                }
            },
        },
        {
            "anyOf": [
                {"type": "string"},
                {
                    "type": "string",
                    "format": "PRIVATE_UNSELECTED_FORMAT_CANARY",
                },
            ]
        },
        {
            "type": "array",
            "items": {
                "type": "string",
                "format": "PRIVATE_ITEM_FORMAT_CANARY",
            },
        },
    ],
    ids=["optional-nested-property", "unselected-anyof-branch", "array-items"],
)
def test_catalog_construction_audits_formats_in_all_nested_schema_nodes(
    property_schema: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter(
            [
                _tool(
                    "synthetic__validate_value",
                    properties={"optional": property_schema},
                )
            ]
        )

    assert "PRIVATE_" not in str(raised.value)


def test_catalog_construction_rejects_declared_format_without_registered_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(Draft202012Validator.FORMAT_CHECKER.checkers, "duration")

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$"):
        AppWorldSemanticAdapter([_tool("synthetic__validate_value")])


def test_catalog_construction_copies_the_full_supported_draft202012_format_set() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "contacts__validate_email",
                properties={"value": {"type": "string", "format": "email"}},
            ),
            _tool(
                "text__validate_regex",
                properties={"value": {"type": "string", "format": "regex"}},
            ),
        ]
    )

    validators = object.__getattribute__(adapter, "_validators")
    email_checker = validators["contacts__validate_email"].format_checker
    regex_checker = validators["text__validate_regex"].format_checker
    draft_checkers = Draft202012Validator.FORMAT_CHECKER.checkers

    assert email_checker is not None
    assert regex_checker is not None
    assert email_checker is not regex_checker
    assert email_checker.checkers is not regex_checker.checkers
    assert email_checker.checkers is not draft_checkers
    assert regex_checker.checkers is not draft_checkers
    expected_formats = (
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
    )
    assert tuple(email_checker.checkers) == expected_formats
    assert tuple(regex_checker.checkers) == expected_formats
    assert all(
        email_checker.checkers[name] is draft_checkers[name]
        and regex_checker.checkers[name] is draft_checkers[name]
        for name in expected_formats
    )


def test_catalog_construction_resolves_and_enforces_local_json_pointer_refs() -> None:
    tool = _tool(
        "synthetic__validate_value",
        properties={"value": {"$ref": "#/$defs/timestamp"}},
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["$defs"] = {
        "timestamp": {"type": "string", "format": "date-time"}
    }

    adapter = AppWorldSemanticAdapter([tool])

    assert adapter.surface_to_semantic(
        {
            "name": "synthetic__validate_value",
            "arguments": {"value": "2000-01-01T00:00:00Z"},
        }
    )
    with pytest.raises(ValueError, match=r"^AppWorld surface call is invalid$"):
        adapter.surface_to_semantic(
            {
                "name": "synthetic__validate_value",
                "arguments": {"value": "PRIVATE_INVALID_DATETIME_CANARY"},
            }
        )


def test_catalog_construction_audits_ref_targets_hidden_under_unknown_keys() -> None:
    private_format = "PRIVATE_REFERENCED_FORMAT_CANARY"
    tool = _tool(
        "synthetic__validate_value",
        properties={"value": {"$ref": "#/x-private/target"}},
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["x-private"] = {
        "target": {"type": "string", "format": private_format}
    }

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter([tool])

    assert private_format not in str(raised.value)


@pytest.mark.parametrize(
    "target",
    [{"type": 7}, {"type": "object", "required": "PRIVATE_REQUIRED_CANARY"}],
    ids=["invalid-type", "invalid-required"],
)
def test_catalog_construction_meta_validates_hidden_local_ref_targets(
    target: dict[str, object],
) -> None:
    tool = _tool(
        "synthetic__validate_value",
        properties={"value": {"$ref": "#/x-private/target"}},
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["x-private"] = {"target": target}

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter([tool])

    assert "PRIVATE_" not in str(raised.value)


@pytest.mark.parametrize(
    "reference",
    [
        "#missing-anchor",
        "relative-schema.json#/$defs/value",
        "https://example.invalid/private-schema.json#/$defs/value",
    ],
    ids=["anchor", "relative", "remote"],
)
def test_catalog_construction_rejects_non_pointer_refs_without_retrieval(
    reference: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrievals: list[object] = []

    def record_retrieval(*args: object, **kwargs: object) -> None:
        retrievals.append((args, kwargs))
        raise AssertionError("retrieval must not run")

    monkeypatch.setattr(urllib.request, "urlopen", record_retrieval)
    tool = _tool(
        "synthetic__validate_value",
        properties={"value": {"$ref": reference}},
        required=["value"],
    )

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter([tool])

    assert reference not in str(raised.value)
    assert retrievals == []


@pytest.mark.parametrize("keyword", ["$dynamicRef", "$recursiveRef"])
def test_catalog_construction_rejects_dynamic_reference_keywords(
    keyword: str,
) -> None:
    tool = _tool(
        "synthetic__validate_value",
        properties={"value": {keyword: "#/$defs/value"}},
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["$defs"] = {"value": {"type": "string"}}

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter([tool])

    assert keyword not in str(raised.value)


@pytest.mark.parametrize(
    "reference",
    ["#/$defs/missing", "#/$defs/bad~2escape", "#/x-private-list/3"],
    ids=["missing", "invalid-escape", "array-index-out-of-range"],
)
def test_catalog_construction_rejects_bad_local_json_pointers(reference: str) -> None:
    tool = _tool(
        "synthetic__validate_value",
        properties={"value": {"$ref": reference}},
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["$defs"] = {
        "present": {"type": "string"},
        "bad~2escape": {"type": "string"},
    }
    parameters["x-private-list"] = [{"type": "string"}]

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$"):
        AppWorldSemanticAdapter([tool])


def test_catalog_construction_rejects_percent_encoded_pointer_fragments() -> None:
    reference = "#/x%2Fprivate/target"
    tool = _tool(
        "synthetic__validate_value",
        properties={"value": {"$ref": reference}},
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["x%2Fprivate"] = {"target": {"type": "string"}}

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter([tool])

    assert reference not in str(raised.value)


@pytest.mark.parametrize("identifier_keyword", ["$id", "id"])
def test_catalog_construction_rejects_nested_resource_identifiers(
    identifier_keyword: str,
) -> None:
    private_identifier = "https://example.invalid/PRIVATE_RESOURCE_CANARY.json"
    tool = _tool(
        "synthetic__validate_value",
        properties={
            "value": {
                identifier_keyword: private_identifier,
                "$defs": {"target": {"type": "string"}},
                "$ref": "#/$defs/target",
            }
        },
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["$defs"] = {"target": {"type": "string"}}

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter([tool])

    assert private_identifier not in str(raised.value)


@pytest.mark.parametrize("location", ["root", "nested", "referenced-target"])
def test_catalog_construction_rejects_schema_dialect_overrides(location: str) -> None:
    alternate_dialect = "http://json-schema.org/draft-07/schema#"
    tool = _tool(
        "synthetic__validate_value",
        properties={
            "value": {
                "type": "object",
                "unevaluatedProperties": False,
            }
        },
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    value_schema = properties["value"]
    assert isinstance(value_schema, dict)
    if location == "root":
        parameters["$schema"] = alternate_dialect
    elif location == "nested":
        value_schema["$schema"] = alternate_dialect
    else:
        value_schema.clear()
        value_schema["$ref"] = "#/x-private/target"
        parameters["x-private"] = {
            "target": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "unevaluatedProperties": False,
            }
        }

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$") as raised:
        AppWorldSemanticAdapter([tool])

    assert alternate_dialect not in str(raised.value)


def test_catalog_construction_rejects_cyclic_local_refs() -> None:
    tool = _tool(
        "synthetic__validate_value",
        properties={"value": {"$ref": "#/$defs/cycle"}},
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["$defs"] = {"cycle": {"$ref": "#/$defs/cycle"}}

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$"):
        AppWorldSemanticAdapter([tool])


def test_catalog_validators_use_independent_closed_registries() -> None:
    adapter = AppWorldSemanticAdapter(
        [_tool("alpha__empty"), _tool("beta__empty")]
    )

    validators = tuple(object.__getattribute__(adapter, "_validators").values())
    registries = tuple(
        object.__getattribute__(validator, "_registry") for validator in validators
    )

    assert all(type(registry) is Registry and len(registry) == 0 for registry in registries)
    assert registries[0] is not registries[1]


def _minimal_source_calls(
    adapter: AppWorldSemanticAdapter,
) -> tuple[Mapping[str, object], ...]:
    return appworld_adapter_module.build_minimal_source_calls(adapter)


def test_build_minimal_source_calls_is_exported_from_adapters() -> None:
    module_builder = getattr(appworld_adapter_module, "build_minimal_source_calls", None)
    package_builder = getattr(adapters_module, "build_minimal_source_calls", None)

    assert callable(module_builder)
    assert package_builder is module_builder


def test_build_minimal_source_calls_covers_scalar_const_and_enum_schemas() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__primitive_values",
                properties={
                    "constant": {"const": "fixed"},
                    "choice": {"type": "integer", "enum": [3, 1, 2]},
                    "flag": {"type": "boolean"},
                    "nothing": {"type": "null"},
                    "optional": {
                        "type": "string",
                        "default": "PRIVATE_DEFAULT_CANARY",
                    },
                },
                required=["constant", "choice", "flag", "nothing"],
            )
        ]
    )

    calls = _minimal_source_calls(adapter)

    assert calls == (
        {
            "name": "synthetic__primitive_values",
            "arguments": {
                "choice": 1,
                "constant": "fixed",
                "flag": False,
                "nothing": None,
            },
        },
    )
    assert "optional" not in calls[0]["arguments"]
    with pytest.raises(TypeError):
        calls[0]["name"] = "forbidden"


def test_build_minimal_source_calls_covers_composition_objects_and_arrays() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__composite_values",
                properties={
                    "any_choice": {
                        "anyOf": [
                            {"type": "string", "minLength": 2},
                            {"type": "integer", "minimum": 5},
                        ]
                    },
                    "one_choice": {
                        "oneOf": [{"type": "boolean"}, {"type": "null"}]
                    },
                    "nullable": {"type": ["string", "null"]},
                    "nested": {
                        "type": "object",
                        "properties": {
                            "required_flag": {"type": "boolean"},
                            "optional_text": {"type": "string", "default": "unused"},
                        },
                        "required": ["required_flag"],
                        "additionalProperties": False,
                    },
                    "items": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 2},
                        "minItems": 2,
                        "maxItems": 3,
                    },
                },
                required=["items", "nested", "nullable", "one_choice", "any_choice"],
            )
        ]
    )

    call = _minimal_source_calls(adapter)[0]

    assert call == {
        "name": "synthetic__composite_values",
        "arguments": {
            "any_choice": 5,
            "items": (2, 2),
            "nested": {"required_flag": False},
            "nullable": None,
            "one_choice": None,
        },
    }


@pytest.mark.parametrize(
    ("branches", "expected"),
    [
        (
            [
                {"type": "string", "minLength": 0},
                {"type": "string", "maxLength": 0},
            ],
            "a",
        ),
        (
            [
                {"type": "array", "items": {"type": "null"}, "minItems": 0},
                {"type": "array", "items": {"type": "null"}, "maxItems": 0},
            ],
            (None,),
        ),
    ],
    ids=["adjacent-string", "adjacent-array"],
)
def test_build_minimal_source_calls_tries_bounded_adjacent_one_of_candidates(
    branches: list[dict[str, object]],
    expected: object,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__exclusive_choice",
                properties={"value": {"oneOf": branches}},
                required=["value"],
            )
        ]
    )

    assert _minimal_source_calls(adapter)[0]["arguments"]["value"] == expected


@pytest.mark.parametrize(
    ("branches", "expected"),
    [
        (
            [
                {
                    "type": "object",
                    "properties": {"x": {"type": "boolean"}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "boolean"},
                        "y": {"type": "boolean"},
                    },
                    "required": ["x"],
                    "additionalProperties": False,
                },
            ],
            {"x": False, "y": False},
        ),
        (
            [
                {"type": "array", "items": {"type": "boolean"}, "minItems": 1},
                {"type": "array", "items": {"const": False}, "minItems": 1},
            ],
            (True,),
        ),
    ],
    ids=["single-optional-property", "alternate-uniform-item"],
)
def test_build_minimal_source_calls_tries_bounded_structural_one_of_neighbors(
    branches: list[dict[str, object]],
    expected: object,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__structural_choice",
                properties={"value": {"oneOf": branches}},
                required=["value"],
            )
        ]
    )

    assert _minimal_source_calls(adapter)[0]["arguments"]["value"] == expected


@pytest.mark.parametrize(
    ("branches", "expected"),
    [
        (
            [
                {"type": "integer", "minimum": 0},
                {"type": "integer", "minimum": 0, "maximum": 100},
            ],
            101,
        ),
        (
            [
                {"type": "string", "minLength": 0},
                {"type": "string", "minLength": 0, "maxLength": 5},
            ],
            "aaaaaa",
        ),
        (
            [
                {"type": "number", "minimum": 0},
                {"type": "number", "minimum": 0, "maximum": 100},
            ],
            math.nextafter(100.0, math.inf),
        ),
    ],
    ids=["integer-upper-neighbor", "string-upper-neighbor", "number-upper-neighbor"],
)
def test_build_minimal_source_calls_uses_cross_branch_boundary_neighbors(
    branches: list[dict[str, object]],
    expected: object,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__boundary_choice",
                properties={"value": {"oneOf": branches}},
                required=["value"],
            )
        ]
    )

    assert _minimal_source_calls(adapter)[0]["arguments"]["value"] == expected


@pytest.mark.parametrize(
    ("branches", "expected"),
    [
        (
            [
                {"type": "array", "items": {"type": "null"}, "minItems": 0},
                {
                    "type": "array",
                    "items": {"type": "null"},
                    "minItems": 0,
                    "maxItems": 100,
                },
            ],
            (None,) * 101,
        ),
        (
            [
                {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 1,
                    "maxItems": 1,
                },
                {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 100},
                    "minItems": 1,
                    "maxItems": 1,
                },
            ],
            (101,),
        ),
        (
            [
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer", "minimum": 0}},
                    "required": ["x"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "minimum": 0, "maximum": 100}
                    },
                    "required": ["x"],
                    "additionalProperties": False,
                },
            ],
            {"x": 101},
        ),
    ],
    ids=["array-length", "array-item", "object-property"],
)
def test_build_minimal_source_calls_preserves_nested_raw_boundary_neighbors(
    branches: list[dict[str, object]],
    expected: object,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__nested_boundary_choice",
                properties={"value": {"oneOf": branches}},
                required=["value"],
            )
        ]
    )

    assert _minimal_source_calls(adapter)[0]["arguments"]["value"] == expected


def test_build_minimal_source_calls_covers_strings_formats_and_ignores_defaults() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__string_values",
                properties={
                    "fixed_length": {
                        "type": "string",
                        "minLength": 3,
                        "maxLength": 3,
                    },
                    "email": {
                        "type": "string",
                        "format": "email",
                        "minLength": 5,
                        "maxLength": 40,
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "minLength": 10,
                        "maxLength": 10,
                    },
                    "required_with_default": {
                        "type": "string",
                        "default": "PRIVATE_REQUIRED_DEFAULT_CANARY",
                    },
                },
                required=["required_with_default", "date", "email", "fixed_length"],
            )
        ]
    )

    arguments = _minimal_source_calls(adapter)[0]["arguments"]

    assert arguments == {
        "date": "2000-01-01",
        "email": "a@example.com",
        "fixed_length": "aaa",
        "required_with_default": "",
    }


def test_build_minimal_source_calls_tries_a_short_supported_format_candidate() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__short_email",
                properties={
                    "value": {"type": "string", "format": "email", "maxLength": 3}
                },
                required=["value"],
            )
        ]
    )

    value = _minimal_source_calls(adapter)[0]["arguments"]["value"]

    assert isinstance(value, str)
    assert len(value) <= 3


@pytest.mark.parametrize(
    ("format_name", "expected"),
    [
        ("date", "2000-01-01"),
        ("date-time", "2000-01-01T00:00:00Z"),
        ("duration", "P0D"),
        ("email", "@"),
        ("hostname", "a"),
        ("idn-email", "@"),
        ("idn-hostname", "a"),
        ("ipv4", "0.0.0.0"),
        ("ipv6", "::"),
        ("iri", "a:b"),
        ("iri-reference", ""),
        ("json-pointer", ""),
        ("regex", ""),
        ("relative-json-pointer", "0"),
        ("time", "00:00:00Z"),
        ("uri", "a:b"),
        ("uri-reference", ""),
        ("uri-template", ""),
        ("uuid", "00000000-0000-0000-0000-000000000000"),
    ],
)
def test_build_minimal_source_calls_covers_every_admitted_format(
    format_name: str,
    expected: str,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__formatted_value",
                properties={"value": {"type": "string", "format": format_name}},
                required=["value"],
            )
        ]
    )

    assert _minimal_source_calls(adapter)[0]["arguments"]["value"] == expected


@pytest.mark.parametrize(
    ("format_name", "length"),
    [("email", 100), ("uri", 4), ("ipv4", 11), ("hostname", 64)],
)
def test_build_minimal_source_calls_covers_bounded_exact_format_lengths(
    format_name: str,
    length: int,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__exact_format_length",
                properties={
                    "value": {
                        "type": "string",
                        "format": format_name,
                        "minLength": length,
                        "maxLength": length,
                    }
                },
                required=["value"],
            )
        ]
    )

    value = _minimal_source_calls(adapter)[0]["arguments"]["value"]

    assert isinstance(value, str)
    assert len(value) == length


def test_build_minimal_source_calls_covers_numeric_bounds_and_multiples() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__numeric_values",
                properties={
                    "positive_integer": {
                        "type": "integer",
                        "exclusiveMinimum": 4,
                        "maximum": 10,
                        "multipleOf": 3,
                    },
                    "negative_integer": {
                        "type": "integer",
                        "exclusiveMaximum": -2,
                        "multipleOf": 2,
                    },
                    "decimal": {
                        "type": "number",
                        "minimum": 0.3,
                        "maximum": 1,
                        "multipleOf": 0.2,
                    },
                    "exclusive_decimal": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 0.5,
                        "multipleOf": 0.2,
                    },
                },
                required=[
                    "positive_integer",
                    "negative_integer",
                    "decimal",
                    "exclusive_decimal",
                ],
            )
        ]
    )

    arguments = _minimal_source_calls(adapter)[0]["arguments"]

    assert arguments == {
        "decimal": 0.4,
        "exclusive_decimal": 0.2,
        "negative_integer": -4,
        "positive_integer": 6,
    }


def test_build_minimal_source_calls_covers_float_multiple_runtime_lattice() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__float_lattice",
                properties={
                    "value": {
                        "type": "number",
                        "exclusiveMinimum": -0.8,
                        "maximum": -0.6,
                        "multipleOf": 0.1,
                    }
                },
                required=["value"],
            )
        ]
    )

    value = _minimal_source_calls(adapter)[0]["arguments"]["value"]

    assert value == pytest.approx(-0.7)


def test_build_minimal_source_calls_checks_float_neighbors_at_exclusive_bounds() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__exclusive_float_lattice",
                properties={
                    "value": {
                        "type": "number",
                        "exclusiveMinimum": 2.4,
                        "exclusiveMaximum": 2.7,
                        "multipleOf": 0.3,
                    }
                },
                required=["value"],
            )
        ]
    )

    value = _minimal_source_calls(adapter)[0]["arguments"]["value"]

    assert value == pytest.approx(2.7)
    assert value < 2.7


def test_build_minimal_source_calls_is_order_independent_and_catalog_complete() -> None:
    first = AppWorldSemanticAdapter(
        [
            _tool("zeta__empty"),
            _tool(
                "alpha__ordered",
                properties={
                    "enum_value": {"type": "integer", "enum": [3, 1, 2]},
                    "branch_value": {
                        "anyOf": [{"type": "string"}, {"type": "boolean"}]
                    },
                },
                required=["enum_value", "branch_value"],
            ),
        ]
    )
    second = AppWorldSemanticAdapter(
        [
            _tool(
                "alpha__ordered",
                properties={
                    "branch_value": {
                        "anyOf": [{"type": "boolean"}, {"type": "string"}]
                    },
                    "enum_value": {"type": "integer", "enum": [2, 3, 1]},
                },
                required=["branch_value", "enum_value"],
            ),
            _tool("zeta__empty"),
        ]
    )

    first_calls = _minimal_source_calls(first)
    second_calls = _minimal_source_calls(second)

    assert canonical_json_bytes(first_calls) == canonical_json_bytes(second_calls)
    assert tuple(call["name"] for call in first_calls) == (
        "alpha__ordered",
        "zeta__empty",
    )


@pytest.mark.parametrize(
    "property_schema",
    [
        {"allOf": [{"type": "string"}]},
        {"type": "string", "pattern": "^PRIVATE_PATTERN_CANARY$"},
        {"type": ["string", "integer"]},
        {"type": ["null", "string", "integer"]},
        {"anyOf": [{"type": "string"}], "oneOf": [{"type": "string"}]},
        {"type": "object", "minProperties": 1},
        {"type": "string", "PRIVATE_ASSERTION_CANARY": True},
    ],
    ids=[
        "all-of",
        "pattern",
        "non-null-union",
        "wide-null-union",
        "mixed-composition",
        "unsupported-object-assertion",
        "unknown-assertion",
    ],
)
def test_build_minimal_source_calls_rejects_unsupported_grammar_without_payload(
    property_schema: dict[str, object],
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "private__unprobeable",
                properties={"value": property_schema},
                required=["value"],
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"^AppWorld catalog is unprobeable$",
    ) as raised:
        _minimal_source_calls(adapter)

    assert "PRIVATE_" not in str(raised.value)
    assert "private__unprobeable" not in str(raised.value)


def test_build_minimal_source_calls_rejects_local_refs_supported_at_construction() -> None:
    tool = _tool(
        "private__referenced",
        properties={"value": {"$ref": "#/$defs/value"}},
        required=["value"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["$defs"] = {"value": {"type": "string"}}
    adapter = AppWorldSemanticAdapter([tool])

    with pytest.raises(
        ValueError,
        match=r"^AppWorld catalog is unprobeable$",
    ) as raised:
        _minimal_source_calls(adapter)

    assert "private__referenced" not in str(raised.value)


@pytest.mark.parametrize(
    "property_schema",
    [
        {"type": "integer", "enum": [1], "minimum": 2},
        {"oneOf": [{"type": "boolean"}, {"type": "boolean"}]},
        {"type": "string", "minLength": 3, "maxLength": 2},
        {"type": "array", "items": {"type": "null"}, "minItems": 2, "maxItems": 1},
        {"type": "number", "minimum": 5, "maximum": 4},
        {"type": "integer", "const": True},
        {"anyOf": [False, False]},
    ],
    ids=[
        "enum-conflict",
        "overlapping-one-of",
        "string-bounds",
        "array-bounds",
        "numeric-bounds",
        "bool-is-not-integer",
        "false-branches",
    ],
)
def test_build_minimal_source_calls_rejects_unsatisfiable_schemas_without_payload(
    property_schema: dict[str, object],
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "private__unsatisfiable",
                properties={"value": property_schema},
                required=["value"],
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"^AppWorld catalog is unprobeable$",
    ) as raised:
        _minimal_source_calls(adapter)

    assert "private__unsatisfiable" not in str(raised.value)


@pytest.mark.parametrize(
    "property_schema",
    [
        {"type": "string", "minLength": 4097},
        {"type": "array", "items": {"type": "null"}, "minItems": 257},
    ],
    ids=["bounded-string", "bounded-array"],
)
def test_build_minimal_source_calls_fails_closed_instead_of_unbounded_generation(
    property_schema: dict[str, object],
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__bounded",
                properties={"value": property_schema},
                required=["value"],
            )
        ]
    )

    with pytest.raises(ValueError, match=r"^AppWorld catalog is unprobeable$"):
        _minimal_source_calls(adapter)


@pytest.mark.parametrize(
    "property_schema",
    [
        {"const": "a" * 4097},
        {"const": [None] * 257},
        {"const": {f"key_{index}": None for index in range(257)}},
        {"enum": ["a" * 4097]},
        {"type": "integer", "minimum": 1, "maximum": 91, "multipleOf": 0.07},
    ],
    ids=[
        "const-string-budget",
        "const-array-budget",
        "const-object-budget",
        "enum-value-budget",
        "fractional-integer-multiple",
    ],
)
def test_build_minimal_source_calls_statically_rejects_out_of_budget_values(
    property_schema: dict[str, object],
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__bounded_explicit_value",
                properties={"value": property_schema},
                required=["value"],
            )
        ]
    )

    with pytest.raises(ValueError, match=r"^AppWorld catalog is unprobeable$"):
        _minimal_source_calls(adapter)


def test_build_minimal_source_calls_generates_each_nested_schema_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_depth = 16
    property_schema: dict[str, object] = {"type": "null"}
    for _ in range(nested_depth):
        property_schema = {
            "type": "array",
            "items": property_schema,
            "minItems": 1,
            "maxItems": 1,
        }
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__nested_single_pass",
                properties={"value": property_schema},
                required=["value"],
            )
        ]
    )
    original = appworld_adapter_module._witness_candidate_set
    calls = 0

    def counted_candidate_set(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        appworld_adapter_module,
        "_witness_candidate_set",
        counted_candidate_set,
    )

    result = _minimal_source_calls(adapter)

    assert len(result) == 1
    assert calls == nested_depth + 2  # root object, arrays, and null leaf


def test_build_minimal_source_calls_rejects_projected_array_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__projected_array_budget",
                properties={
                    "value": {
                        "type": "array",
                        "items": {"type": "null"},
                        "minItems": 3,
                        "maxItems": 3,
                    }
                },
                required=["value"],
            )
        ]
    )
    events: list[tuple[str, int]] = []
    original_project = appworld_adapter_module._project_uniform_array_budget
    original_materialize = appworld_adapter_module._materialize_uniform_array

    def project(item: object, length: int) -> object:
        events.append(("project", length))
        return original_project(item, length)

    def materialize(item: object, length: int) -> object:
        events.append(("materialize", length))
        return original_materialize(item, length)

    monkeypatch.setattr(appworld_adapter_module, "_MAX_WITNESS_VALUE_NODES", 3)
    monkeypatch.setattr(
        appworld_adapter_module,
        "_project_uniform_array_budget",
        project,
    )
    monkeypatch.setattr(
        appworld_adapter_module,
        "_materialize_uniform_array",
        materialize,
    )

    with pytest.raises(ValueError, match=r"^AppWorld catalog is unprobeable$"):
        _minimal_source_calls(adapter)

    assert events == [("project", 3)]


def test_build_minimal_source_calls_rejects_projected_object_before_member_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__projected_object_budget",
                properties={
                    "value": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "null"},
                            "b": {"type": "null"},
                        },
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    }
                },
                required=["value"],
            )
        ]
    )
    events: list[tuple[str, str]] = []
    original_project = appworld_adapter_module._project_object_candidate_budget
    original_materialize = appworld_adapter_module._materialize_object_candidate

    def project(
        values: Mapping[str, object],
        name: str,
        value: object,
    ) -> object:
        events.append(("project", name))
        return original_project(values, name, value)

    def materialize(
        values: Mapping[str, object],
        name: str,
        value: object,
    ) -> object:
        events.append(("materialize", name))
        return original_materialize(values, name, value)

    monkeypatch.setattr(appworld_adapter_module, "_MAX_WITNESS_VALUE_NODES", 2)
    monkeypatch.setattr(
        appworld_adapter_module,
        "_project_object_candidate_budget",
        project,
    )
    monkeypatch.setattr(
        appworld_adapter_module,
        "_materialize_object_candidate",
        materialize,
    )

    with pytest.raises(ValueError, match=r"^AppWorld catalog is unprobeable$"):
        _minimal_source_calls(adapter)

    assert events == [
        ("project", "a"),
        ("materialize", "a"),
        ("project", "b"),
    ]


def test_build_minimal_source_calls_bounds_composition_pool_while_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__bounded_composition_pool",
                properties={
                    "value": {
                        "oneOf": [{"const": str(index)} for index in range(8)]
                    }
                },
                required=["value"],
            )
        ]
    )
    original = appworld_adapter_module._witness_candidate_set
    calls = 0

    def counted_candidate_set(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(appworld_adapter_module, "_MAX_WITNESS_CANDIDATE_POOL_BYTES", 7)
    monkeypatch.setattr(
        appworld_adapter_module,
        "_witness_candidate_set",
        counted_candidate_set,
    )

    with pytest.raises(ValueError, match=r"^AppWorld catalog is unprobeable$"):
        _minimal_source_calls(adapter)

    assert calls == 5  # root, oneOf, then only three of eight branches


def test_build_minimal_source_calls_uses_a_fresh_format_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__email",
                properties={"value": {"type": "string", "format": "email"}},
                required=["value"],
            )
        ]
    )
    monkeypatch.setitem(
        Draft202012Validator.FORMAT_CHECKER.checkers,
        "email",
        (lambda _value: False, ()),
    )

    with pytest.raises(ValueError, match=r"^AppWorld catalog is unprobeable$"):
        _minimal_source_calls(adapter)


def test_build_minimal_source_calls_confirms_each_call_through_the_source_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AppWorldSemanticAdapter([_tool("synthetic__empty")])
    original = AppWorldSemanticAdapter.surface_to_semantic
    confirmed: list[object] = []

    def confirm(
        self: AppWorldSemanticAdapter,
        surface_call: Mapping[str, object],
    ) -> tuple[SemanticAction, ...]:
        confirmed.append(surface_call)
        return original(self, surface_call)  # type: ignore[arg-type]

    monkeypatch.setattr(AppWorldSemanticAdapter, "surface_to_semantic", confirm)

    calls = _minimal_source_calls(adapter)

    assert confirmed == list(calls)


def test_build_minimal_source_calls_preserves_integrity_error_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__email",
                properties={"value": {"type": "string", "format": "email"}},
                required=["value"],
            )
        ]
    )
    binding = object.__getattribute__(adapter, "_bindings")["synthetic__email"]

    def mutate(_value: object) -> bool:
        object.__setattr__(binding, "api_name", "PRIVATE_MUTATION_CANARY")
        return True

    monkeypatch.setitem(
        Draft202012Validator.FORMAT_CHECKER.checkers,
        "email",
        (mutate, ()),
    )

    with pytest.raises(
        ValueError,
        match=r"^AppWorld adapter integrity validation failed$",
    ) as raised:
        _minimal_source_calls(adapter)

    assert "PRIVATE_MUTATION_CANARY" not in str(raised.value)


def test_build_minimal_source_calls_rechecks_all_tools_after_confirmation_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AppWorldSemanticAdapter(
        [_tool("alpha__empty"), _tool("zeta__empty")]
    )
    original = AppWorldSemanticAdapter.surface_to_semantic
    alpha_binding = object.__getattribute__(adapter, "_bindings")["alpha__empty"]

    def mutate_prior_binding(
        self: AppWorldSemanticAdapter,
        surface_call: Mapping[str, object],
    ) -> tuple[SemanticAction, ...]:
        actions = original(self, surface_call)  # type: ignore[arg-type]
        if surface_call["name"] == "zeta__empty":
            object.__setattr__(alpha_binding, "api_name", "PRIVATE_PRIOR_CANARY")
        return actions

    monkeypatch.setattr(
        AppWorldSemanticAdapter,
        "surface_to_semantic",
        mutate_prior_binding,
    )

    with pytest.raises(
        ValueError,
        match=r"^AppWorld adapter integrity validation failed$",
    ) as raised:
        _minimal_source_calls(adapter)

    assert "PRIVATE_PRIOR_CANARY" not in str(raised.value)


def test_build_minimal_source_calls_rejects_the_whole_catalog_on_one_failure() -> None:
    adapter = AppWorldSemanticAdapter(
        [
            _tool("alpha__valid"),
            _tool(
                "private__invalid",
                properties={"value": {"type": "string", "pattern": "PRIVATE_CANARY"}},
                required=["value"],
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"^AppWorld catalog is unprobeable$",
    ) as raised:
        _minimal_source_calls(adapter)

    assert "alpha__valid" not in str(raised.value)
    assert "private__invalid" not in str(raised.value)


def test_importing_appworld_adapter_does_not_import_external_appworld() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import toolshift.adapters.appworld; "
                "assert not any(name == 'appworld' or name.startswith('appworld.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_surface_to_semantic_parses_one_exact_native_call() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    call = _event_call()

    actions = adapter.surface_to_semantic(call)  # type: ignore[arg-type]

    assert type(actions) is tuple
    assert len(actions) == 1
    assert actions[0] == SemanticAction(
        "calendar__create_event",
        {
            "title": "Synthetic meeting",
            "attendees": ["synthetic@example.com"],
            "options": {"notify": True},
        },
    )
    assert set(call) == {"name", "arguments"}


def test_surface_to_semantic_snapshots_nested_arguments() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    call = _event_call()
    arguments = call["arguments"]
    assert isinstance(arguments, dict)

    action = adapter.surface_to_semantic(call)[0]  # type: ignore[arg-type]
    arguments["title"] = "changed"

    assert action.arguments["title"] == "Synthetic meeting"
    assert action.arguments["attendees"] == ("synthetic@example.com",)
    assert action.arguments["options"] == {"notify": True}
    with pytest.raises(TypeError):
        action.arguments["title"] = "forbidden"  # type: ignore[index]


def test_surface_to_semantic_does_not_apply_schema_defaults() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())

    action = adapter.surface_to_semantic(_event_call())[0]  # type: ignore[arg-type]

    assert "retries" not in action.arguments


@pytest.mark.parametrize(
    ("call", "private_canary"),
    [
        ({"arguments": {}}, "missing-name-canary"),
        ({"name": "mail__send_message"}, "missing-arguments-canary"),
        ({"name": 7, "arguments": {}}, "wrong-name-canary"),
        ({"name": "mail__send_message", "arguments": []}, "array-arguments-canary"),
        (
            {"name": "mail__send_message", "arguments": {}},
            "missing-required-canary",
        ),
        (
            {
                "name": "mail__send_message",
                "arguments": {"body": "ok", "private_extra": "PRIVATE_EXTRA_CANARY"},
            },
            "PRIVATE_EXTRA_CANARY",
        ),
        (
            {"name": "mail__send_message", "arguments": {"body": 7}},
            "wrong-type-canary",
        ),
        (
            {
                "name": "calendar__create_event",
                "arguments": {
                    "title": "Synthetic meeting",
                    "attendees": ["synthetic@example.com"],
                    "options": {"notify": True},
                    "retries": True,
                },
            },
            "bool-as-int-canary",
        ),
        (
            {"name": "private__unknown", "arguments": {"secret": "PRIVATE_TOOL_CANARY"}},
            "PRIVATE_TOOL_CANARY",
        ),
        (
            {
                "name": "mail__send_message",
                "arguments": {"body": "ok"},
                "metadata": {"secret": "PRIVATE_METADATA_CANARY"},
            },
            "PRIVATE_METADATA_CANARY",
        ),
        (
            {
                "name": "mail__send_message",
                "arguments": {"body": "ok"},
                "metadata": float("nan"),
            },
            "nan-metadata-canary",
        ),
        (
            {
                "name": "mail__send_message",
                "arguments": {"body": "ok", "unsafe": 2**53},
            },
            "unsafe-integer-canary",
        ),
    ],
    ids=[
        "missing-name",
        "missing-arguments",
        "wrong-name-type",
        "wrong-arguments-type",
        "missing-required",
        "extra-argument",
        "wrong-argument-type",
        "bool-is-not-integer",
        "unknown-tool",
        "extra-call-metadata",
        "non-ijson-metadata",
        "non-ijson-argument",
    ],
)
def test_surface_to_semantic_rejects_invalid_calls_without_payload(
    call: dict[str, object],
    private_canary: str,
) -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())

    with pytest.raises(ValueError, match=r"^AppWorld surface call is invalid$") as raised:
        adapter.surface_to_semantic(call)  # type: ignore[arg-type]

    assert private_canary not in str(raised.value)


def test_surface_to_semantic_rejects_cyclic_arguments_without_payload() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    cyclic: dict[str, object] = {}
    cyclic["PRIVATE_CYCLE_CANARY"] = cyclic

    with pytest.raises(ValueError, match=r"^AppWorld surface call is invalid$") as raised:
        adapter.surface_to_semantic(
            {"name": "mail__send_message", "arguments": cyclic}  # type: ignore[arg-type]
        )

    assert "PRIVATE_CYCLE_CANARY" not in str(raised.value)


def test_semantic_to_base_calls_compiles_one_exact_immutable_native_call() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    action = adapter.surface_to_semantic(
        _event_call()  # type: ignore[arg-type]
    )[0]

    base_calls = adapter.semantic_to_base_calls(action)

    assert type(base_calls) is tuple
    assert len(base_calls) == 1
    assert base_calls[0] == {
        "name": "calendar__create_event",
        "arguments": {
            "title": "Synthetic meeting",
            "attendees": ("synthetic@example.com",),
            "options": {"notify": True},
        },
    }
    assert set(base_calls[0]) == {"name", "arguments"}
    with pytest.raises(TypeError):
        base_calls[0]["name"] = "forbidden"  # type: ignore[index]


@pytest.mark.parametrize(
    "action",
    [
        SemanticAction("private__unknown", {"secret": "PRIVATE_ACTION_CANARY"}),
        SemanticAction("mail__send_message", {}),
        SemanticAction("mail__send_message", {"body": 7}),
    ],
    ids=["unknown-tool", "missing-required", "wrong-type"],
)
def test_semantic_to_base_calls_rejects_invalid_actions_without_payload(
    action: SemanticAction,
) -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())

    with pytest.raises(ValueError, match=r"^AppWorld semantic action is invalid$") as raised:
        adapter.semantic_to_base_calls(action)

    assert "PRIVATE_ACTION_CANARY" not in str(raised.value)


def test_semantic_to_base_calls_rejects_tampered_action_snapshot() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    action = SemanticAction("mail__send_message", {"body": "Synthetic"})
    object.__setattr__(action, "_arguments_canonical", b"PRIVATE_ACTION_SNAPSHOT_CANARY")

    with pytest.raises(ValueError, match=r"^AppWorld semantic action is invalid$") as raised:
        adapter.semantic_to_base_calls(action)

    assert "PRIVATE_ACTION_SNAPSHOT_CANARY" not in str(raised.value)


def test_base_observation_to_surface_wraps_one_deeply_frozen_observation() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    call = _event_call()
    action = adapter.surface_to_semantic(call)[0]  # type: ignore[arg-type]
    observation: dict[str, object] = {
        "event": {"created": True, "attendees": ["synthetic@example.com"]}
    }

    surface = adapter.base_observation_to_surface(
        call,  # type: ignore[arg-type]
        (action,),
        ((observation,),),  # type: ignore[arg-type]
    )
    observation["event"] = {"created": False}

    assert surface == {
        "event": {"created": True, "attendees": ("synthetic@example.com",)}
    }
    assert isinstance(surface, Mapping)
    with pytest.raises(TypeError):
        surface["event"] = None  # type: ignore[index]


@pytest.mark.parametrize(
    ("actions_factory", "groups"),
    [
        (lambda parsed: (), ()),
        (lambda parsed: (parsed, parsed), ((None,), (None,))),
        (lambda parsed: (SemanticAction("mail__send_message", {"body": "other"}),), ((None,),)),
        (lambda parsed: (parsed,), ()),
        (lambda parsed: (parsed,), ((),)),
        (lambda parsed: (parsed,), ((None, None),)),
    ],
    ids=[
        "no-action",
        "multiple-actions",
        "mismatched-action",
        "no-group",
        "empty-group",
        "multiple-observations",
    ],
)
def test_base_observation_to_surface_rejects_invalid_grouping_without_payload(
    actions_factory: Callable[[SemanticAction], tuple[SemanticAction, ...]],
    groups: tuple[tuple[object, ...], ...],
) -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    call = {"name": "mail__send_message", "arguments": {"body": "Synthetic"}}
    parsed = adapter.surface_to_semantic(call)[0]

    with pytest.raises(ValueError, match=r"^AppWorld observation group is invalid$") as raised:
        adapter.base_observation_to_surface(
            call,
            actions_factory(parsed),
            groups,  # type: ignore[arg-type]
        )

    assert "Synthetic" not in str(raised.value)


def test_base_observation_to_surface_rejects_non_ijson_observation() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    call = {"name": "mail__send_message", "arguments": {"body": "Synthetic"}}
    parsed = adapter.surface_to_semantic(call)[0]

    with pytest.raises(ValueError, match=r"^AppWorld observation group is invalid$") as raised:
        adapter.base_observation_to_surface(
            call,
            (parsed,),
            (({"secret": object()},),),  # type: ignore[arg-type]
        )

    assert "secret" not in str(raised.value)
    assert "object" not in str(raised.value)


def test_canonicalize_trace_independently_reparses_exact_native_calls() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    surface_calls = (
        _event_call(),
        {"name": "mail__send_message", "arguments": {"body": "Synthetic"}},
    )
    actions = tuple(
        adapter.surface_to_semantic(call)[0]  # type: ignore[arg-type]
        for call in surface_calls
    )
    base_calls = tuple(adapter.semantic_to_base_calls(action)[0] for action in actions)
    trace = ExecutionTrace(surface_calls, actions, base_calls)  # type: ignore[arg-type]

    canonical = adapter.canonicalize_trace(trace)

    assert canonical == actions
    assert set(base_calls[0]) == {"name", "arguments"}


@pytest.mark.parametrize(
    "mutation",
    [
        "reordered-actions",
        "extra-action",
        "missing-action",
        "reordered-base-calls",
        "extra-base-call",
        "missing-base-call",
        "hybrid-base-call",
        "tampered-surface-snapshot",
        "tampered-base-snapshot",
        "tampered-action-snapshot",
    ],
)
def test_canonicalize_trace_rejects_inconsistent_or_tampered_channels(
    mutation: str,
) -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    surface_calls = (
        _event_call(),
        {"name": "mail__send_message", "arguments": {"body": "Synthetic"}},
    )
    actions = tuple(
        adapter.surface_to_semantic(call)[0]  # type: ignore[arg-type]
        for call in surface_calls
    )
    base_calls = tuple(adapter.semantic_to_base_calls(action)[0] for action in actions)

    if mutation == "reordered-actions":
        trace = ExecutionTrace(surface_calls, tuple(reversed(actions)), base_calls)  # type: ignore[arg-type]
    elif mutation == "extra-action":
        trace = ExecutionTrace(surface_calls, (*actions, actions[0]), base_calls)  # type: ignore[arg-type]
    elif mutation == "missing-action":
        trace = ExecutionTrace(surface_calls, actions[:-1], base_calls)  # type: ignore[arg-type]
    elif mutation == "reordered-base-calls":
        trace = ExecutionTrace(surface_calls, actions, tuple(reversed(base_calls)))  # type: ignore[arg-type]
    elif mutation == "extra-base-call":
        trace = ExecutionTrace(surface_calls, actions, (*base_calls, base_calls[0]))  # type: ignore[arg-type]
    elif mutation == "missing-base-call":
        trace = ExecutionTrace(surface_calls, actions, base_calls[:-1])  # type: ignore[arg-type]
    elif mutation == "hybrid-base-call":
        hybrid = dict(base_calls[0])
        hybrid["metadata"] = "PRIVATE_HYBRID_CANARY"
        trace = ExecutionTrace(surface_calls, actions, (hybrid, base_calls[1]))  # type: ignore[arg-type]
    else:
        trace = ExecutionTrace(surface_calls, actions, base_calls)  # type: ignore[arg-type]
        if mutation == "tampered-surface-snapshot":
            object.__setattr__(
                trace,
                "_surface_calls_canonical",
                b"PRIVATE_SURFACE_SNAPSHOT_CANARY",
            )
        elif mutation == "tampered-base-snapshot":
            object.__setattr__(
                trace,
                "_base_calls_canonical",
                b"PRIVATE_BASE_SNAPSHOT_CANARY",
            )
        else:
            object.__setattr__(
                actions[0],
                "_arguments_canonical",
                b"PRIVATE_ACTION_SNAPSHOT_CANARY",
            )

    with pytest.raises(ValueError, match=r"^AppWorld execution trace is invalid$") as raised:
        adapter.canonicalize_trace(trace)

    assert "PRIVATE_" not in str(raised.value)


def test_canonicalize_trace_accepts_an_empty_consistent_trace() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())

    assert adapter.canonicalize_trace(ExecutionTrace((), (), ())) == ()


class _MutatingRuntimeMapping(Mapping[str, object]):
    def __init__(
        self,
        entries: Mapping[str, object],
        mutation: Callable[[], None],
    ) -> None:
        self._entries = dict(entries)
        self._mutation = mutation

    def __getitem__(self, key: str) -> object:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self):
        self._mutation()
        return self._entries.items()


def _tamper_adapter(adapter: AppWorldSemanticAdapter, target: str) -> None:
    bindings = object.__getattribute__(adapter, "_bindings")
    validators = object.__getattribute__(adapter, "_validators")
    binding = bindings["calendar__create_event"]
    validator = validators["calendar__create_event"]
    if target == "variant-slot":
        object.__setattr__(adapter, "_variant", object())
    elif target == "variant-name":
        object.__setattr__(adapter.variant.tools[0], "name", "private__variant_canary")
    elif target == "bindings-root":
        object.__setattr__(adapter, "_bindings", {})
    elif target == "binding-name":
        object.__setattr__(binding, "api_name", "private_binding_canary")
    elif target == "binding-schema":
        object.__setattr__(binding, "schema", {})
    elif target == "binding-schema-canonical":
        object.__setattr__(binding, "schema_canonical", b"PRIVATE_SCHEMA_CANARY")
    elif target == "validators-root":
        object.__setattr__(adapter, "_validators", {})
    elif target == "validator-schema":
        object.__setattr__(validator, "schema", {})
    elif target == "validator-format-checker":
        object.__setattr__(validator, "format_checker", FormatChecker())
    elif target == "validator-checkers":
        validator.format_checker.checkers.clear()
    else:  # pragma: no cover - keeps the helper fail-closed if extended incorrectly.
        raise AssertionError("unknown synthetic tamper target")


@pytest.mark.parametrize(
    "target",
    [
        "variant-slot",
        "variant-name",
        "bindings-root",
        "binding-name",
        "binding-schema",
        "binding-schema-canonical",
        "validators-root",
        "validator-schema",
        "validator-format-checker",
        "validator-checkers",
    ],
)
@pytest.mark.parametrize("method", ["parse", "compile", "wrap", "trace"])
def test_semantic_methods_reject_preexisting_adapter_tampering(
    target: str,
    method: str,
) -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    call = _event_call()
    action = adapter.surface_to_semantic(call)[0]  # type: ignore[arg-type]
    base_call = adapter.semantic_to_base_calls(action)[0]
    trace = ExecutionTrace((call,), (action,), (base_call,))  # type: ignore[arg-type]
    operations = {
        "parse": lambda: adapter.surface_to_semantic(call),  # type: ignore[arg-type]
        "compile": lambda: adapter.semantic_to_base_calls(action),
        "wrap": lambda: adapter.base_observation_to_surface(
            call,  # type: ignore[arg-type]
            (action,),
            ((None,),),
        ),
        "trace": lambda: adapter.canonicalize_trace(trace),
    }
    _tamper_adapter(adapter, target)

    with pytest.raises(
        ValueError,
        match=r"^AppWorld adapter integrity validation failed$",
    ) as raised:
        operations[method]()

    assert "private" not in str(raised.value).lower()


@pytest.mark.parametrize("phase", ["surface-call", "observation"])
def test_callback_mutation_is_detected_by_final_integrity_check(phase: str) -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())

    def mutate() -> None:
        _tamper_adapter(adapter, "binding-name")

    if phase == "surface-call":
        operation = lambda: adapter.surface_to_semantic(  # noqa: E731
            _MutatingRuntimeMapping(_event_call(), mutate)  # type: ignore[arg-type]
        )
    else:
        call = _event_call()
        action = adapter.surface_to_semantic(call)[0]  # type: ignore[arg-type]
        observation = _MutatingRuntimeMapping({"ok": True}, mutate)
        operation = lambda: adapter.base_observation_to_surface(  # noqa: E731
            call,  # type: ignore[arg-type]
            (action,),
            ((observation,),),  # type: ignore[arg-type]
        )

    with pytest.raises(
        ValueError,
        match=r"^AppWorld adapter integrity validation failed$",
    ):
        operation()


def test_mapping_callback_errors_are_sanitized() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())

    class ExplodingMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            return iter(())

        def __len__(self) -> int:
            return 0

        def items(self):
            raise ValueError("PRIVATE_CALLBACK_CANARY")

    with pytest.raises(ValueError, match=r"^AppWorld surface call is invalid$") as raised:
        adapter.surface_to_semantic(ExplodingMapping())  # type: ignore[arg-type]

    assert "PRIVATE_CALLBACK_CANARY" not in str(raised.value)


@pytest.mark.parametrize("method", ["parse", "compile", "wrap", "trace", "witness"])
def test_public_appworld_operations_sanitize_forged_integrity_callback_errors(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    private_message = "PRIVATE_FORGED_INTEGRITY_CANARY"

    def forge_integrity(_value: object) -> bool:
        raise appworld_adapter_module._AdapterIntegrityError(private_message)

    monkeypatch.setitem(
        Draft202012Validator.FORMAT_CHECKER.checkers,
        "email",
        (forge_integrity, ()),
    )
    adapter = AppWorldSemanticAdapter(
        [
            _tool(
                "synthetic__email",
                properties={"value": {"type": "string", "format": "email"}},
                required=["value"],
            )
        ]
    )
    call = {"name": "synthetic__email", "arguments": {"value": "a@example.com"}}
    action = SemanticAction("synthetic__email", {"value": "a@example.com"})
    trace = ExecutionTrace((call,), (action,), (call,))  # type: ignore[arg-type]
    operations = {
        "parse": lambda: adapter.surface_to_semantic(call),
        "compile": lambda: adapter.semantic_to_base_calls(action),
        "wrap": lambda: adapter.base_observation_to_surface(
            call,
            (action,),
            ((None,),),
        ),
        "trace": lambda: adapter.canonicalize_trace(trace),
        "witness": lambda: _minimal_source_calls(adapter),
    }

    with pytest.raises(
        ValueError,
        match=r"^AppWorld adapter integrity validation failed$",
    ) as raised:
        operations[method]()

    assert private_message not in str(raised.value)


def test_runtime_seal_container_replacement_is_rejected() -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    seals = object.__getattribute__(adapter, "_binding_runtime_seals")
    equal_replacement = tuple(list(seals))
    assert equal_replacement == seals
    assert equal_replacement is not seals
    object.__setattr__(adapter, "_binding_runtime_seals", equal_replacement)

    with pytest.raises(
        ValueError,
        match=r"^AppWorld adapter integrity validation failed$",
    ):
        adapter.surface_to_semantic(_event_call())  # type: ignore[arg-type]


def test_hot_path_checks_only_the_selected_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AppWorldSemanticAdapter(_semantic_catalog())
    validators = object.__getattribute__(adapter, "_validators")
    unselected = validators["mail__send_message"]
    original = appworld_adapter_module._validator_seal_matches

    def selected_only(seal: object) -> bool:
        if seal.validator is unselected:  # type: ignore[attr-defined]
            raise AssertionError("unselected validator visited")
        return original(seal)  # type: ignore[arg-type]

    monkeypatch.setattr(
        appworld_adapter_module,
        "_validator_seal_matches",
        selected_only,
    )

    action = adapter.surface_to_semantic(_event_call())[0]  # type: ignore[arg-type]

    assert action.name == "calendar__create_event"


def test_catalog_order_does_not_change_the_source_variant() -> None:
    first = build_appworld_adapter(_catalog())
    second = build_appworld_adapter(list(reversed(_catalog())))

    assert first.variant == second.variant
    assert canonical_json_bytes(first.variant.manifest) == canonical_json_bytes(
        second.variant.manifest
    )


def test_build_appworld_adapter_accepts_mapping_catalog_entries() -> None:
    class ReadOnlyMapping(Mapping[str, object]):
        def __init__(self, value: Mapping[str, object]) -> None:
            self._value = dict(value)
            self.reads = 0

        def __getitem__(self, key: str) -> object:
            self.reads += 1
            return self._value[key]

        def __iter__(self) -> Iterator[str]:
            return iter(tuple(self._value))

        def __len__(self) -> int:
            return len(self._value)

    entry = ReadOnlyMapping(_tool("notes__create_note"))

    adapter = build_appworld_adapter((entry,))

    assert entry.reads > 0
    assert adapter.variant.tools[0].name == "notes__create_note"


def test_adapter_variant_is_detached_from_caller_mutation() -> None:
    catalog = _catalog()
    adapter = build_appworld_adapter(catalog)

    function = catalog[0]["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    properties["body"] = {"type": "number"}
    catalog.clear()

    send = adapter.variant.tools[1]
    assert send.input_schema["properties"]["body"]["type"] == "string"


@pytest.mark.parametrize(
    "catalog",
    [
        [],
        "not-a-catalog",
        ["not-a-tool"],
        [{"type": "function"}],
        [{"type": "function", "function": {}, "extra": True}],
        [
            {
                "type": "not-function",
                "function": {
                    "name": "notes__create_note",
                    "description": "Synthetic.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ],
        [
            {
                "type": "function",
                "function": {
                    "name": "notes__create_note",
                    "description": "Synthetic.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "extra": True,
                },
            }
        ],
    ],
    ids=[
        "empty",
        "text-sequence",
        "non-mapping-entry",
        "missing-function-fields",
        "outer-extra-field",
        "wrong-type-tag",
        "function-extra-field",
    ],
)
def test_build_appworld_adapter_rejects_invalid_catalog_shapes(catalog: Any) -> None:
    with pytest.raises(ValueError, match="catalog"):
        build_appworld_adapter(catalog)


@pytest.mark.parametrize(
    "name",
    [
        "missing_separator",
        "__missing_app",
        "missing_api__",
        "app-name__api",
        "app__api-name",
        "应用__api",
        "app__api name",
    ],
)
def test_build_appworld_adapter_rejects_invalid_native_names(name: str) -> None:
    with pytest.raises(ValueError, match="catalog") as raised:
        build_appworld_adapter([_tool(name)])

    assert name not in str(raised.value)


def test_build_appworld_adapter_rejects_duplicate_names_without_payload() -> None:
    private_name = "private__duplicate"

    with pytest.raises(ValueError, match="catalog") as raised:
        build_appworld_adapter([_tool(private_name), _tool(private_name)])

    assert private_name not in str(raised.value)


@pytest.mark.parametrize(
    "reserved_name",
    [
        "_app_name",
        "_api_name",
        "client",
        "raise_on_failure",
        "show",
        "track",
        "_system_datetime",
    ],
)
def test_build_appworld_adapter_rejects_requester_control_properties_without_payload(
    reserved_name: str,
) -> None:
    tool = _tool(
        "synthetic__operation",
        properties={reserved_name: {"type": "string"}},
    )

    with pytest.raises(ValueError, match="schema") as raised:
        build_appworld_adapter([tool])

    assert reserved_name not in str(raised.value)


def test_build_appworld_adapter_rejects_root_pattern_properties_control_bypass() -> None:
    private_pattern = "^track$"
    tool = _tool("synthetic__operation")
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["patternProperties"] = {
        private_pattern: {"type": "boolean"},
    }

    with pytest.raises(ValueError, match="schema") as raised:
        build_appworld_adapter([tool])

    assert private_pattern not in str(raised.value)


@pytest.mark.parametrize(
    ("schema_update", "private_canary"),
    [
        ({"type": "array"}, "array-root-canary"),
        ({"properties": []}, "properties-canary"),
        ({"required": "private-required"}, "private-required"),
        ({"required": ["missing-private-property"]}, "missing-private-property"),
        ({"required": ["value", "value"]}, "duplicate-required-canary"),
        ({"additionalProperties": True}, "open-root-canary"),
        ({"additionalProperties": {}}, "schema-open-root-canary"),
    ],
)
def test_build_appworld_adapter_rejects_invalid_root_schemas_without_payload(
    schema_update: dict[str, object],
    private_canary: str,
) -> None:
    tool = _tool("synthetic__operation", properties={"value": {"type": "string"}})
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters.update(schema_update)

    with pytest.raises(ValueError, match="schema") as raised:
        build_appworld_adapter([tool])

    assert private_canary not in str(raised.value)


def test_build_appworld_adapter_rejects_invalid_json_schema_without_payload() -> None:
    private_canary = "private-invalid-json-schema-type"
    tool = _tool(
        "synthetic__operation",
        properties={"value": {"type": private_canary}},
    )

    with pytest.raises(ValueError, match="schema") as raised:
        build_appworld_adapter([tool])

    assert private_canary not in str(raised.value)


def test_build_appworld_adapter_requires_exact_required_list() -> None:
    tool = _tool(
        "synthetic__operation",
        properties={"first": {"type": "string"}, "second": {"type": "string"}},
        required=["second", "first"],
    )
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    parameters["required"] = ("second", "first")

    with pytest.raises(ValueError, match="schema"):
        build_appworld_adapter([tool])


def test_build_appworld_adapter_preserves_required_order() -> None:
    adapter = build_appworld_adapter(
        [
            _tool(
                "synthetic__operation",
                properties={
                    "first": {"type": "string"},
                    "second": {"type": "string"},
                },
                required=["second", "first"],
            )
        ]
    )

    assert adapter.variant.tools[0].input_schema["required"] == ("second", "first")


def test_build_appworld_adapter_rejects_non_json_catalog_values_without_payload() -> None:
    private_payload = object()
    tool = _tool(
        "synthetic__operation",
        properties={"value": {"type": "string", "default": private_payload}},
    )

    with pytest.raises(ValueError, match="catalog") as raised:
        build_appworld_adapter([tool])

    assert repr(private_payload) not in str(raised.value)
