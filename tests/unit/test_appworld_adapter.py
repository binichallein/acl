"""Synthetic-only tests for the pure AppWorld semantic adapter."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker

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


def test_catalog_only_adapter_fails_closed_for_all_semantic_methods() -> None:
    adapter = AppWorldSemanticAdapter(_catalog())
    private_call = {"name": "private__canary", "arguments": {"secret": "private-value"}}
    action = SemanticAction("private__canary", {"secret": "private-value"})
    trace = ExecutionTrace((private_call,), (action,), (private_call,))
    operations = (
        lambda: adapter.surface_to_semantic(private_call),
        lambda: adapter.semantic_to_base_calls(action),
        lambda: adapter.base_observation_to_surface(private_call, (action,), ((None,),)),
        lambda: adapter.canonicalize_trace(trace),
    )

    for operation in operations:
        with pytest.raises(ValueError, match="semantic behavior is unavailable") as raised:
            operation()
        assert "private" not in str(raised.value)


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
