"""Contract tests for the canonical semantic layer."""

from __future__ import annotations

from collections.abc import Hashable, ItemsView, Mapping, ValuesView
from dataclasses import FrozenInstanceError
from datetime import datetime
from inspect import signature
from pathlib import Path

import pytest
import yaml

import toolshift.types as types_module
from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.schema import schema_fingerprint
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    canonical_json_bytes,
    manifest_sha256,
)


def _tool(name: str = "search") -> SurfaceToolSpec:
    return SurfaceToolSpec(
        name=name,
        description="Search for a document",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


def _variant(*names: str) -> SchemaVariant:
    tool_names = names or ("search",)
    return SchemaVariant(
        variant_id="identity-v1",
        tools=[_tool(name) for name in tool_names],
        manifest={"operator": "identity", "seed": 7},
    )


def _nested_list(depth: int, leaf: JSONValue = "leaf") -> JSONValue:
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


class _IdentityAdapter(SemanticAdapter):
    """Small contract implementation kept in tests, not production code."""

    def _require_known_action(self, action: SemanticAction) -> None:
        if not isinstance(action, SemanticAction):
            raise ValueError("action must be a SemanticAction")
        if action.name not in {tool.name for tool in self.variant.tools}:
            raise ValueError(f"unknown action: {action.name}")

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        if not isinstance(surface_call, Mapping):
            raise ValueError("surface_call must be a mapping")
        try:
            name = surface_call["name"]
            arguments = surface_call["arguments"]
        except KeyError as error:
            raise ValueError(f"surface_call missing field: {error.args[0]}") from error
        if not isinstance(name, str) or name not in {tool.name for tool in self.variant.tools}:
            raise ValueError(f"unknown surface tool: {name!r}")
        if not isinstance(arguments, Mapping):
            raise ValueError("surface_call.arguments must be a mapping")
        return (SemanticAction(name=name, arguments=arguments),)

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        self._require_known_action(action)
        return ({"name": action.name, "arguments": action.arguments},)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        if not isinstance(actions, tuple) or len(actions) != 1:
            raise ValueError("identity adapter requires exactly one semantic action")
        if not isinstance(base_observation_groups, tuple) or len(base_observation_groups) != 1:
            raise ValueError("identity adapter requires exactly one observation group")
        expected_actions = self.surface_to_semantic(surface_call)
        if actions != expected_actions:
            raise ValueError("actions do not match the surface call")
        self._require_known_action(actions[0])
        group = base_observation_groups[0]
        if not isinstance(group, tuple) or len(group) != 1:
            raise ValueError("identity action requires exactly one base observation")
        return group[0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        if not isinstance(trace, ExecutionTrace):
            raise ValueError("trace must be an ExecutionTrace")
        for action in trace.semantic_actions:
            self._require_known_action(action)
        return trace.semantic_actions


class _MergeAdapter(SemanticAdapter):
    """Test-only counterexample where one surface call expands to two actions."""

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        if (
            not isinstance(surface_call, Mapping)
            or surface_call.get("name") != "schedule_and_notify"
        ):
            raise ValueError("unknown or malformed surface call")
        arguments = surface_call.get("arguments")
        if not isinstance(arguments, Mapping):
            raise ValueError("surface_call.arguments must be a mapping")
        try:
            title = arguments["title"]
            recipient = arguments["recipient"]
            text = arguments["text"]
        except KeyError as error:
            raise ValueError(f"surface_call missing argument: {error.args[0]}") from error
        return (
            SemanticAction("create_event", {"title": title}),
            SemanticAction("send_message", {"recipient": recipient, "text": text}),
        )

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        if action.name == "create_event":
            return ({"name": "calendar.create", "arguments": action.arguments},)
        if action.name == "send_message":
            return ({"name": "messaging.send", "arguments": action.arguments},)
        raise ValueError(f"unknown action: {action.name}")

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        expected_actions = self.surface_to_semantic(surface_call)
        if actions != expected_actions:
            raise ValueError("actions do not match the surface call")
        if len(actions) != len(base_observation_groups):
            raise ValueError("actions and base observation groups must have equal length")
        if len(actions) != 2 or any(len(group) != 1 for group in base_observation_groups):
            raise ValueError("merge adapter requires two singleton observation groups")
        return {
            "event": base_observation_groups[0][0],
            "notification": base_observation_groups[1][0],
        }

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        if not isinstance(trace, ExecutionTrace):
            raise ValueError("trace must be an ExecutionTrace")
        return trace.semantic_actions


def test_semantic_action_preserves_name_arguments_and_nested_order() -> None:
    action = SemanticAction(
        name="  search  ",
        arguments={"queries": ["first", "second"], "options": {"limit": 2}},
    )

    assert action.name == "  search  "
    assert action.arguments["queries"] == ("first", "second")
    assert action.arguments["options"]["limit"] == 2


def test_surface_tool_spec_allows_empty_description_and_preserves_text() -> None:
    tool = SurfaceToolSpec(
        name=" lookup ",
        description="",
        input_schema={"required": ["query", "limit"]},
    )

    assert tool.name == " lookup "
    assert tool.description == ""
    assert tool.input_schema["required"] == ("query", "limit")


def test_surface_tool_schema_preserves_nested_wide_integers_and_fingerprint() -> None:
    wide = 2**80 + 123
    source_schema = {
        "type": "object",
        "default": {"metadata": {"identifiers": [wide, -wide]}},
        "properties": {
            "count": {
                "type": "integer",
                "maximum": wide,
                "examples": [wide],
            }
        },
    }
    first_tool = SurfaceToolSpec("count", "", source_schema)
    reordered_tool = SurfaceToolSpec(
        "count",
        "",
        {
            "properties": {
                "count": {
                    "examples": [wide],
                    "maximum": wide,
                    "type": "integer",
                }
            },
            "default": {"metadata": {"identifiers": [wide, -wide]}},
            "type": "object",
        },
    )
    first_variant = SchemaVariant("wide-schema", [first_tool], {"version": 1})
    reordered_variant = SchemaVariant(
        "wide-schema",
        [reordered_tool],
        {"version": 1},
    )

    source_schema["properties"]["count"]["maximum"] = 0
    source_schema["default"]["metadata"]["identifiers"][0] = 0

    assert first_tool.input_schema["properties"]["count"]["maximum"] == wide
    assert first_tool.input_schema["properties"]["count"]["examples"] == (wide,)
    assert first_tool.input_schema["default"]["metadata"]["identifiers"] == (
        wide,
        -wide,
    )
    assert schema_fingerprint(first_variant) == schema_fingerprint(reordered_variant)
    with pytest.raises(ValueError, match="I-JSON safe range"):
        canonical_json_bytes(first_tool.input_schema)
    with pytest.raises(TypeError):
        first_tool.input_schema["properties"]["count"]["maximum"] = 0  # type: ignore[index]


def test_surface_tool_schema_accepts_integers_beyond_python_decimal_guard() -> None:
    very_wide = 10**5000
    tool = SurfaceToolSpec(
        "count",
        "",
        {"type": "object", "maximum": very_wide},
    )
    variant = SchemaVariant("very-wide-schema", [tool], {"version": 1})

    assert tool.input_schema["maximum"] == very_wide
    assert len(schema_fingerprint(variant)) == 64


def test_surface_tool_schema_keeps_json_safety_guards_for_non_integer_values() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    for schema in (
        {"bad": float("inf")},
        {"bad": "\ud800"},
        cyclic,
        {"nested": _nested_list(128)},
    ):
        with pytest.raises(ValueError, match=r"input_schema|cycle|depth|UTF-8"):
            SurfaceToolSpec("guarded", "", schema)  # type: ignore[arg-type]


def test_schema_fingerprint_rejects_tampered_wide_integer_schema() -> None:
    wide = 2**80 + 123
    tool = SurfaceToolSpec(
        "count",
        "",
        {"type": "object", "maximum": wide},
    )
    replacement = SurfaceToolSpec(
        "count",
        "",
        {"type": "object", "maximum": wide + 1},
    )
    variant = SchemaVariant("wide-schema", [tool], {"version": 1})
    replacement_items = object.__getattribute__(replacement.input_schema, "_items")

    object.__setattr__(tool.input_schema, "_items", replacement_items)

    with pytest.raises(ValueError, match="mutated"):
        schema_fingerprint(variant)


@pytest.mark.parametrize("name", ["", " ", "\t\n"])
def test_names_and_variant_ids_must_not_be_blank(name: str) -> None:
    with pytest.raises(ValueError, match="name"):
        SemanticAction(name=name, arguments={})
    with pytest.raises(ValueError, match="name"):
        SurfaceToolSpec(name=name, description="", input_schema={})
    with pytest.raises(ValueError, match="variant_id"):
        SchemaVariant(variant_id=name, tools=[_tool()], manifest={})


def test_names_descriptions_and_ids_require_strings() -> None:
    with pytest.raises(ValueError, match="name"):
        SemanticAction(name=1, arguments={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="description"):
        SurfaceToolSpec(
            name="search",
            description=None,  # type: ignore[arg-type]
            input_schema={},
        )
    with pytest.raises(ValueError, match="variant_id"):
        SchemaVariant(variant_id=3, tools=[_tool()], manifest={})  # type: ignore[arg-type]


def test_schema_variant_requires_tools_with_unique_names() -> None:
    with pytest.raises(ValueError, match="at least one tool"):
        SchemaVariant(variant_id="empty", tools=[], manifest={})
    with pytest.raises(ValueError, match="unique"):
        SchemaVariant(
            variant_id="duplicate",
            tools=[_tool("search"), _tool("search")],
            manifest={},
        )
    with pytest.raises(ValueError, match="tools"):
        SchemaVariant(
            variant_id="wrong-type",
            tools=[_tool(), "not-a-tool"],  # type: ignore[list-item]
            manifest={},
        )


def test_all_semantic_dataclasses_take_deep_immutable_snapshots() -> None:
    action_arguments = {"query": "alpha", "filters": ["new"]}
    input_schema = {"properties": {"query": {"type": "string"}}}
    manifest = {"mapping": {"search": "lookup"}, "order": ["rename"]}
    surface_call = {"name": "lookup", "arguments": {"query": "alpha"}}
    base_call = {"name": "search", "arguments": {"query": "alpha"}}

    action = SemanticAction("search", action_arguments)
    tool = SurfaceToolSpec("lookup", "", input_schema)
    tools = [tool]
    variant = SchemaVariant("rename-v1", tools, manifest)
    surface_calls = [surface_call]
    semantic_actions = [action]
    base_calls = [base_call]
    trace = ExecutionTrace(surface_calls, semantic_actions, base_calls)
    action_snapshot = canonical_json_bytes(action.arguments)
    variant_digest = manifest_sha256(variant.manifest)
    trace_snapshot = canonical_json_bytes(trace.surface_calls)

    action_arguments["query"] = "changed"
    action_arguments["filters"].append("old")
    input_schema["properties"]["query"]["type"] = "integer"
    manifest["mapping"]["search"] = "changed"
    manifest["order"].append("restructure")
    tools.append(_tool("open"))
    surface_call["name"] = "changed"
    surface_calls.append({"name": "extra"})
    semantic_actions.clear()
    base_call["name"] = "changed"
    base_calls.clear()

    assert canonical_json_bytes(action.arguments) == action_snapshot
    assert manifest_sha256(variant.manifest) == variant_digest
    assert canonical_json_bytes(trace.surface_calls) == trace_snapshot
    assert tool.input_schema["properties"]["query"]["type"] == "string"
    assert len(variant.tools) == 1
    assert trace.semantic_actions == (action,)
    assert trace.base_calls[0]["name"] == "search"

    with pytest.raises(FrozenInstanceError):
        action.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        action.arguments["query"] = "changed"  # type: ignore[index]
    with pytest.raises(AttributeError):
        action.arguments["filters"].append("changed")  # type: ignore[union-attr]


@pytest.mark.parametrize("operation", ["assign", "delete"])
def test_frozen_json_mapping_rejects_slot_mutation(operation: str) -> None:
    arguments = SemanticAction("search", {"query": "alpha"}).arguments
    attribute = "_items"

    with pytest.raises(AttributeError):
        if operation == "assign":
            setattr(arguments, attribute, ())
        else:
            delattr(arguments, attribute)


def test_frozen_json_mapping_methods_reject_noncanonical_raw_items() -> None:
    arguments = SemanticAction("search", {"query": "alpha"}).arguments

    class CallbackTuple(tuple[object, ...]):
        fired = False

        def __iter__(self):
            self.fired = True
            return super().__iter__()

    injected = CallbackTuple((("query", "changed"),))
    object.__setattr__(arguments, "_items", injected)

    with pytest.raises(ValueError, match="state"):
        dict(arguments)
    assert injected.fired is False


def test_frozen_json_mapping_full_iteration_avoids_per_key_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping_type = type(SemanticAction("seed", {}).arguments)
    original_getitem = mapping_type.__getitem__
    lookup_calls = 0

    def counted_getitem(mapping: object, key: str) -> JSONValue:
        nonlocal lookup_calls
        lookup_calls += 1
        return original_getitem(mapping, key)

    monkeypatch.setattr(mapping_type, "__getitem__", counted_getitem)

    arguments = SemanticAction(
        "search",
        {f"key-{index:04d}": index for index in range(256)},
    ).arguments
    assert len(tuple(arguments.items())) == 256
    assert len(tuple(arguments.values())) == 256
    assert canonical_json_bytes(arguments).startswith(b'{"key-0000":0')
    assert lookup_calls == 0


def test_frozen_json_mapping_length_does_not_scan_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = SemanticAction(
        "search",
        {f"key-{index:04d}": index for index in range(256)},
    ).arguments
    mapping_type = type(arguments)
    original_raw_items = mapping_type._raw_items
    scan_calls = 0

    def counted_raw_items(mapping: object) -> tuple[tuple[str, JSONValue], ...]:
        nonlocal scan_calls
        scan_calls += 1
        return original_raw_items(mapping)

    monkeypatch.setattr(mapping_type, "_raw_items", counted_raw_items)

    assert len(arguments) == 256
    assert scan_calls == 0


def test_frozen_json_mapping_dict_conversion_performs_one_linear_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {f"key-{index:04d}": index for index in range(256)}
    arguments = SemanticAction("search", expected).arguments
    mapping_type = type(arguments)
    original_getitem = mapping_type.__getitem__
    original_raw_items = mapping_type._raw_items
    lookup_calls = 0
    scan_calls = 0

    def counted_getitem(mapping: object, key: str) -> JSONValue:
        nonlocal lookup_calls
        lookup_calls += 1
        return original_getitem(mapping, key)

    def counted_raw_items(mapping: object) -> tuple[tuple[str, JSONValue], ...]:
        nonlocal scan_calls
        scan_calls += 1
        return original_raw_items(mapping)

    monkeypatch.setattr(mapping_type, "__getitem__", counted_getitem)
    monkeypatch.setattr(mapping_type, "_raw_items", counted_raw_items)

    assert dict(arguments) == expected
    assert lookup_calls == len(expected)
    assert scan_calls == 1


def test_frozen_json_mapping_hides_and_rejects_tampered_lookup_index() -> None:
    arguments = SemanticAction(
        "search",
        {"query": "alpha", "limit": 2},
    ).arguments

    with pytest.raises(AttributeError):
        _ = arguments._index  # type: ignore[attr-defined]

    index = object.__getattribute__(arguments, "_index")
    assert type(index) is dict
    index["query"] = 1

    assert types_module._is_frozen_mapping(arguments) is False
    with pytest.raises(ValueError, match="state"):
        _ = arguments["query"]


def test_frozen_json_mapping_shape_check_does_not_invoke_tampered_index_keys() -> None:
    arguments = SemanticAction("search", {"query": "alpha"}).arguments

    class CallbackKey(str):
        fired = False

        def __hash__(self) -> int:
            self.fired = True
            return super().__hash__()

        def __eq__(self, other: object) -> bool:
            self.fired = True
            return super().__eq__(other)

    callback_key = CallbackKey("query")
    index = object.__getattribute__(arguments, "_index")
    index.clear()
    index[callback_key] = 0
    callback_key.fired = False

    assert types_module._is_frozen_mapping(arguments) is False
    assert callback_key.fired is False


def test_frozen_json_mapping_shape_check_does_not_call_rebindable_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = SemanticAction("search", {"query": "alpha"}).arguments
    mapping_type = type(arguments)
    method_called = False

    def callback(_: object) -> tuple[tuple[str, JSONValue], ...]:
        nonlocal method_called
        method_called = True
        raise AssertionError("shape validation invoked a rebound method")

    monkeypatch.setattr(mapping_type, "_raw_items", callback)

    assert types_module._is_frozen_mapping(arguments) is True
    assert method_called is False


def test_frozen_json_mapping_preserves_standard_view_behavior() -> None:
    arguments = SemanticAction(
        "search",
        {"query": "alpha", "limit": 2},
    ).arguments
    items = arguments.items()
    values = arguments.values()

    assert isinstance(items, ItemsView)
    assert isinstance(values, ValuesView)
    assert items == {("query", "alpha"), ("limit", 2)}
    assert list(values) == ["alpha", 2]
    with pytest.raises(TypeError):
        _ = items[0]  # type: ignore[index]


def test_frozen_json_mapping_repr_is_stable_readable_and_address_free() -> None:
    arguments = SemanticAction(
        "search",
        {"query": "alpha", "options": {"limit": 2}},
    ).arguments

    assert repr(arguments) == (
        "FrozenJSONMapping({'query': 'alpha', 'options': FrozenJSONMapping({'limit': 2})})"
    )
    assert "0x" not in repr(arguments)


@pytest.mark.parametrize(
    ("value", "context"),
    [
        ({1: "not a string key"}, "key"),
        ({"bad": {"set"}}, "bad"),
        ({"bad": b"bytes"}, "bad"),
        ({"bad": Path("artifact")}, "bad"),
        ({"bad": datetime(2026, 8, 30)}, "bad"),
        ({"bad": float("nan")}, "bad"),
        ({"bad": float("inf")}, "bad"),
        ({"bad": float("-inf")}, "bad"),
    ],
)
def test_public_json_domain_rejects_invalid_values_with_context(
    value: object, context: str
) -> None:
    with pytest.raises(ValueError, match=context):
        canonical_json_bytes(value)  # type: ignore[arg-type]


def test_json_surrogate_values_are_rejected_during_construction_with_context() -> None:
    surrogate = "\ud800"

    with pytest.raises(ValueError, match=r"arguments\.text"):
        SemanticAction("search", {"text": surrogate})
    with pytest.raises(ValueError, match="value"):
        canonical_json_bytes(surrogate)


def test_json_surrogate_mapping_keys_are_rejected_with_key_context() -> None:
    surrogate = "\ud800"

    with pytest.raises(ValueError, match="key"):
        SemanticAction("search", {surrogate: "text"})
    with pytest.raises(ValueError, match="key"):
        canonical_json_bytes({surrogate: "text"})


def test_dataclass_text_fields_require_valid_utf8_without_normalizing() -> None:
    surrogate = "\ud800"

    with pytest.raises(ValueError, match="name"):
        SemanticAction(surrogate, {})
    with pytest.raises(ValueError, match="name"):
        SurfaceToolSpec(surrogate, "", {})
    with pytest.raises(ValueError, match="description"):
        SurfaceToolSpec("search", surrogate, {})
    with pytest.raises(ValueError, match="variant_id"):
        SchemaVariant(surrogate, [_tool()], {})


def test_dataclass_json_fields_reject_invalid_shapes_with_field_context() -> None:
    with pytest.raises(ValueError, match="arguments"):
        SemanticAction("search", ["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input_schema"):
        SurfaceToolSpec("search", "", {"bad": b"bytes"})
    with pytest.raises(ValueError, match="manifest"):
        SchemaVariant("identity", [_tool()], {"bad": float("nan")})
    with pytest.raises(ValueError, match="surface_calls"):
        ExecutionTrace([{"bad": {1, 2}}], [], [])
    with pytest.raises(ValueError, match="semantic_actions"):
        ExecutionTrace([], ["not-an-action"], [])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="base_calls"):
        ExecutionTrace([], [], [{"bad": Path("artifact")}])


def test_recursive_json_containers_raise_value_error_instead_of_recursion_error() -> None:
    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)
    cyclic_mapping: dict[str, object] = {}
    cyclic_mapping["self"] = cyclic_mapping

    with pytest.raises(ValueError, match="cycle"):
        canonical_json_bytes(cyclic_list)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cycle"):
        SemanticAction("search", cyclic_mapping)  # type: ignore[arg-type]


def test_json_container_depth_128_is_allowed_and_129_is_rejected() -> None:
    assert canonical_json_bytes(_nested_list(128)).startswith(b"[")
    action = SemanticAction("search", {"payload": _nested_list(127)})

    assert action.arguments["payload"] is not None
    with pytest.raises(ValueError, match="depth"):
        canonical_json_bytes(_nested_list(129))
    with pytest.raises(ValueError, match="depth"):
        SemanticAction("search", {"payload": _nested_list(128)})


def test_reviewer_depth_495_case_is_rejected_before_equality() -> None:
    nested = _nested_list(495)

    with pytest.raises(ValueError, match="depth"):
        SemanticAction("search", {"payload": nested})
    with pytest.raises(ValueError, match="depth"):
        canonical_json_bytes(nested)


@pytest.mark.parametrize("value", [-(2**53 - 1), 2**53 - 1])
def test_json_safe_integer_boundaries_are_supported(value: int) -> None:
    action = SemanticAction("search", {"cursor": value})

    assert canonical_json_bytes(value) == str(value).encode("ascii")
    assert action.arguments["cursor"] == value


@pytest.mark.parametrize("value", [-(2**53), 2**53, 10**100])
def test_json_integers_outside_ieee754_exact_range_are_rejected(value: int) -> None:
    with pytest.raises(ValueError, match="value"):
        canonical_json_bytes(value)
    with pytest.raises(ValueError, match=r"arguments\.cursor"):
        SemanticAction("search", {"cursor": value})


def test_runtime_trace_and_manifest_still_reject_wide_integers() -> None:
    wide = 2**80 + 123

    with pytest.raises(ValueError, match="manifest"):
        SchemaVariant("identity", [_tool()], {"wide": wide})
    with pytest.raises(ValueError, match="surface_calls"):
        ExecutionTrace(
            ({"name": "search", "arguments": {"wide": wide}},),
            (),
            (),
        )
    with pytest.raises(ValueError, match="base_calls"):
        ExecutionTrace(
            (),
            (),
            ({"name": "search", "arguments": {"wide": wide}},),
        )


def test_json_booleans_remain_valid_and_distinct_from_integers() -> None:
    action = SemanticAction("search", {"enabled": True})

    assert canonical_json_bytes(True) == b"true"
    assert canonical_json_bytes(False) == b"false"
    assert action.arguments["enabled"] is True


def test_execution_trace_preserves_order_without_forcing_equal_cardinality() -> None:
    first = SemanticAction("search", {"query": "alpha"})
    trace = ExecutionTrace(
        surface_calls=[
            {"name": "lookup", "arguments": {"q": "alpha"}},
            {"name": "confirm", "arguments": {}},
        ],
        semantic_actions=[first],
        base_calls=[
            {"name": "search", "arguments": {"query": "alpha"}},
            {"name": "audit", "arguments": {}},
        ],
    )

    assert [call["name"] for call in trace.surface_calls] == ["lookup", "confirm"]
    assert trace.semantic_actions == (first,)
    assert [call["name"] for call in trace.base_calls] == ["search", "audit"]
    assert tuple(map(len, (trace.surface_calls, trace.semantic_actions, trace.base_calls))) == (
        2,
        1,
        2,
    )


def test_semantic_adapter_declares_exactly_four_abstract_interfaces() -> None:
    assert SemanticAdapter.__abstractmethods__ == {
        "surface_to_semantic",
        "semantic_to_base_calls",
        "base_observation_to_surface",
        "canonicalize_trace",
    }


def test_observation_adapter_contract_preserves_surface_and_action_group_identity() -> None:
    parameters = tuple(signature(SemanticAdapter.base_observation_to_surface).parameters)

    assert parameters == (
        "self",
        "surface_call",
        "actions",
        "base_observation_groups",
    )


def test_incomplete_semantic_adapter_cannot_be_instantiated() -> None:
    class IncompleteAdapter(SemanticAdapter):
        def surface_to_semantic(
            self, surface_call: Mapping[str, JSONValue]
        ) -> tuple[SemanticAction, ...]:
            return ()

        def semantic_to_base_calls(
            self, action: SemanticAction
        ) -> tuple[Mapping[str, JSONValue], ...]:
            return ({},)

        def base_observation_to_surface(
            self,
            surface_call: Mapping[str, JSONValue],
            actions: tuple[SemanticAction, ...],
            base_observation_groups: tuple[tuple[JSONValue, ...], ...],
        ) -> JSONValue:
            return None

    with pytest.raises(TypeError, match="abstract"):
        IncompleteAdapter(_variant())


def test_adapter_binds_schema_variant_and_rejects_other_values() -> None:
    variant = _variant()
    adapter = _IdentityAdapter(variant)

    assert adapter.variant is variant
    with pytest.raises(ValueError, match="variant"):
        _IdentityAdapter("identity-v1")  # type: ignore[arg-type]


def test_test_only_identity_adapter_exercises_all_four_interfaces() -> None:
    adapter = _IdentityAdapter(_variant())
    surface_call = {"name": "search", "arguments": {"query": "alpha"}}
    action = adapter.surface_to_semantic(surface_call)[0]
    base_calls = adapter.semantic_to_base_calls(action)
    observation = adapter.base_observation_to_surface(
        surface_call,
        (action,),
        (({"items": [1, 2]},),),
    )
    trace = ExecutionTrace(
        surface_calls=[{"name": "search", "arguments": {"query": "alpha"}}],
        semantic_actions=[action],
        base_calls=base_calls,
    )

    assert action == SemanticAction("search", {"query": "alpha"})
    assert len(base_calls) == 1
    assert base_calls[0]["name"] == "search"
    assert observation == {"items": [1, 2]}
    assert adapter.canonicalize_trace(trace) == (action,)


def test_test_only_merge_adapter_round_trip_keeps_action_group_identity() -> None:
    adapter = _MergeAdapter(_variant("schedule_and_notify"))
    surface_call = {
        "name": "schedule_and_notify",
        "arguments": {
            "title": "Review",
            "recipient": "team",
            "text": "Starts at 10",
        },
    }
    actions = adapter.surface_to_semantic(surface_call)
    base_calls = tuple(adapter.semantic_to_base_calls(action) for action in actions)
    observation_groups = (
        ({"event_id": "evt-7"},),
        ({"message_id": "msg-9"},),
    )

    assert tuple(action.name for action in actions) == ("create_event", "send_message")
    assert tuple(calls[0]["name"] for calls in base_calls) == (
        "calendar.create",
        "messaging.send",
    )
    assert adapter.base_observation_to_surface(
        surface_call,
        actions,
        observation_groups,
    ) == {
        "event": {"event_id": "evt-7"},
        "notification": {"message_id": "msg-9"},
    }


def test_adapter_shape_and_unknown_errors_are_value_errors_not_key_errors() -> None:
    adapter = _IdentityAdapter(_variant())
    unknown = SemanticAction("delete", {})
    surface_call = {"name": "search", "arguments": {}}

    invalid_operations = [
        lambda: adapter.surface_to_semantic([]),  # type: ignore[arg-type]
        lambda: adapter.surface_to_semantic({"name": "search"}),
        lambda: adapter.surface_to_semantic({"name": "delete", "arguments": {}}),
        lambda: adapter.surface_to_semantic({"name": "search", "arguments": []}),
        lambda: adapter.semantic_to_base_calls(unknown),
        lambda: adapter.base_observation_to_surface(surface_call, (unknown,), (({},),)),
        lambda: adapter.base_observation_to_surface(surface_call, (), ()),
        lambda: adapter.base_observation_to_surface(
            surface_call,
            (SemanticAction("search", {}),),
            ((),),
        ),
        lambda: adapter.canonicalize_trace("not-a-trace"),  # type: ignore[arg-type]
    ]

    for operation in invalid_operations:
        with pytest.raises(ValueError):
            operation()


def test_canonical_trace_ignores_surface_and_base_spelling_but_not_action_order() -> None:
    adapter = _IdentityAdapter(_variant("search", "open"))
    search = SemanticAction("search", {"query": "alpha"})
    open_document = SemanticAction("open", {"id": 3})
    first = ExecutionTrace(
        surface_calls=[{"name": "lookup"}, {"name": "read"}],
        semantic_actions=[search, open_document],
        base_calls=[{"api": "search_documents"}, {"api": "open_document"}],
    )
    second = ExecutionTrace(
        surface_calls=[{"tool": "find"}, {"tool": "fetch"}],
        semantic_actions=[search, open_document],
        base_calls=[{"endpoint": "query"}, {"endpoint": "get"}],
    )
    reordered = ExecutionTrace(
        surface_calls=second.surface_calls,
        semantic_actions=[open_document, search],
        base_calls=second.base_calls,
    )

    assert adapter.canonicalize_trace(first) == adapter.canonicalize_trace(second)
    assert adapter.canonicalize_trace(first) != adapter.canonicalize_trace(reordered)


def test_semantic_action_equality_is_json_type_sensitive_at_every_depth() -> None:
    boolean = SemanticAction("search", {"value": True})
    integer = SemanticAction("search", {"value": 1})
    floating = SemanticAction("search", {"value": 1.0})
    nested_boolean = SemanticAction("search", {"nested": [{"value": True}]})
    nested_integer = SemanticAction("search", {"nested": [{"value": 1}]})

    assert boolean != integer
    assert boolean != floating
    assert integer != floating
    assert nested_boolean != nested_integer


def test_dataclass_equality_ignores_mapping_order_but_preserves_array_order() -> None:
    left_action = SemanticAction(
        "search",
        {"options": {"limit": 2, "fresh": True}, "queries": ["a", "b"]},
    )
    reordered_action = SemanticAction(
        "search",
        {"queries": ["a", "b"], "options": {"fresh": True, "limit": 2}},
    )
    reversed_array_action = SemanticAction(
        "search",
        {"options": {"limit": 2, "fresh": True}, "queries": ["b", "a"]},
    )

    assert left_action == reordered_action
    assert left_action != reversed_array_action


def test_near_limit_equality_is_total_and_does_not_call_public_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_action = SemanticAction("search", {"payload": _nested_list(127)})
    right_action = SemanticAction("search", {"payload": _nested_list(127)})
    left_tool = SurfaceToolSpec("search", "", {"payload": _nested_list(127)})
    right_tool = SurfaceToolSpec("search", "", {"payload": _nested_list(127)})
    left_variant = SchemaVariant(
        "variant",
        [_tool()],
        {"payload": _nested_list(127)},
    )
    right_variant = SchemaVariant(
        "variant",
        [_tool()],
        {"payload": _nested_list(127)},
    )
    left_trace = ExecutionTrace(
        [{"payload": _nested_list(126)}],
        [SemanticAction("search", {"query": "alpha"})],
        [{"payload": _nested_list(126)}],
    )
    right_trace = ExecutionTrace(
        [{"payload": _nested_list(126)}],
        [SemanticAction("search", {"query": "alpha"})],
        [{"payload": _nested_list(126)}],
    )

    def unexpected_public_encode(value: JSONValue) -> bytes:
        raise AssertionError(f"equality re-encoded validated value: {type(value).__name__}")

    monkeypatch.setattr(types_module, "canonical_json_bytes", unexpected_public_encode)

    for left, right in (
        (left_action, right_action),
        (left_tool, right_tool),
        (left_variant, right_variant),
        (left_trace, right_trace),
    ):
        assert left is not right
        assert (left == left) is True
        assert (left == right) is True
        assert (right == left) is True


def test_all_four_dataclass_equalities_use_canonical_json_type_semantics() -> None:
    boolean_tool = SurfaceToolSpec(
        "search",
        "",
        {"type": "object", "default": True},
    )
    integer_tool = SurfaceToolSpec(
        "search",
        "",
        {"default": 1, "type": "object"},
    )
    reordered_boolean_tool = SurfaceToolSpec(
        "search",
        "",
        {"default": True, "type": "object"},
    )
    boolean_variant = SchemaVariant(
        "variant",
        [boolean_tool],
        {"mapping": {"enabled": True}, "operator": "identity"},
    )
    integer_variant = SchemaVariant(
        "variant",
        [boolean_tool],
        {"operator": "identity", "mapping": {"enabled": 1}},
    )
    reordered_boolean_variant = SchemaVariant(
        "variant",
        [reordered_boolean_tool],
        {"operator": "identity", "mapping": {"enabled": True}},
    )
    boolean_trace = ExecutionTrace(
        [{"name": "search", "arguments": {"enabled": True}}],
        [SemanticAction("search", {"enabled": True})],
        [{"arguments": {"enabled": True}, "name": "search"}],
    )
    integer_trace = ExecutionTrace(
        [{"arguments": {"enabled": 1}, "name": "search"}],
        [SemanticAction("search", {"enabled": 1})],
        [{"name": "search", "arguments": {"enabled": 1}}],
    )
    reordered_boolean_trace = ExecutionTrace(
        [{"arguments": {"enabled": True}, "name": "search"}],
        [SemanticAction("search", {"enabled": True})],
        [{"name": "search", "arguments": {"enabled": True}}],
    )

    assert boolean_tool != integer_tool
    assert boolean_tool == reordered_boolean_tool
    assert boolean_variant != integer_variant
    assert boolean_variant == reordered_boolean_variant
    assert boolean_trace != integer_trace
    assert boolean_trace == reordered_boolean_trace


def test_all_four_dataclasses_are_explicitly_unhashable() -> None:
    action = SemanticAction("search", {"query": "alpha"})
    tool = _tool()
    variant = _variant()
    trace = ExecutionTrace(
        [{"name": "search", "arguments": {"query": "alpha"}}],
        [action],
        [{"name": "search", "arguments": {"query": "alpha"}}],
    )

    for value in (action, tool, variant, trace):
        assert not isinstance(value, Hashable)
        with pytest.raises(TypeError, match="unhashable"):
            hash(value)


def test_canonical_manifest_bytes_and_sha256_match_fixed_vector() -> None:
    manifest = {
        "composition_order": ["rename", "restructure"],
        "mapping": {"tools": {"search": "lookup"}},
        "operator": "rename",
        "schema_version": 1,
        "seed": 7,
        "version_hash": "abc123",
    }
    expected = (
        b'{"composition_order":["rename","restructure"],'
        b'"mapping":{"tools":{"search":"lookup"}},"operator":"rename",'
        b'"schema_version":1,"seed":7,"version_hash":"abc123"}'
    )

    assert canonical_json_bytes(manifest) == expected
    assert not expected.startswith(b"\xef\xbb\xbf")
    assert not expected.endswith(b"\n")
    assert manifest_sha256(manifest) == (
        "f01255b76ea4a77f8414da1ade5c49631d72a49e096947bbece5d5cc9b5debbf"
    )


def test_nested_python_and_yaml_key_order_have_the_same_digest() -> None:
    python_manifest = {
        "mapping": {"tools": {"search": "lookup", "open": "read"}, "arguments": {"q": "query"}},
        "composition_order": ["rename", "restructure"],
        "seed": 7,
    }
    yaml_manifest = yaml.safe_load(
        """
seed: 7
composition_order:
  - rename
  - restructure
mapping:
  arguments:
    q: query
  tools:
    open: read
    search: lookup
"""
    )
    reordered_composition = dict(python_manifest)
    reordered_composition["composition_order"] = ["restructure", "rename"]

    assert manifest_sha256(python_manifest) == manifest_sha256(yaml_manifest)
    assert manifest_sha256(python_manifest) != manifest_sha256(reordered_composition)


def test_canonical_json_preserves_array_order_numeric_type_and_unicode_text() -> None:
    assert canonical_json_bytes(["a", "b"]) == canonical_json_bytes(("a", "b"))
    assert canonical_json_bytes(["a", "b"]) != canonical_json_bytes(["b", "a"])
    assert canonical_json_bytes(1) != canonical_json_bytes(1.0)
    assert canonical_json_bytes(" caf\u00e9 ") == b'" caf\xc3\xa9 "'
    assert canonical_json_bytes("\u00e9") != canonical_json_bytes("e\u0301")
