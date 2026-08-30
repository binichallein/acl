"""Synthetic-only tests for the pure AppWorld semantic adapter."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
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
    monkeypatch.delitem(Draft202012Validator.FORMAT_CHECKER.checkers, "email")

    with pytest.raises(ValueError, match=r"^AppWorld schema is invalid$"):
        AppWorldSemanticAdapter(
            [
                _tool(
                    "synthetic__validate_value",
                    properties={"value": {"type": "string", "format": "email"}},
                )
            ]
        )


def test_catalog_construction_copies_only_declared_draft202012_format_checkers() -> None:
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
    assert tuple(email_checker.checkers) == ("email",)
    assert tuple(regex_checker.checkers) == ("regex",)
    assert email_checker.checkers["email"] is draft_checkers["email"]
    assert regex_checker.checkers["regex"] is draft_checkers["regex"]


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
