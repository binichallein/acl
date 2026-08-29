"""Schema behavioral-equivalence contract tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import pytest

from toolshift.contracts.schema import (
    ContractDiagnostic,
    LayerContractResult,
    SchemaProbe,
    check_schema_contract,
    schema_fingerprint,
)
from toolshift.types import (
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    canonical_json_bytes,
)

from ._support import (
    _AlwaysEqualDigest,
    _BadSchemaAdapter,
    _ExplodingAdapter,
    _IdentityAdapter,
    _NondeterministicAdapter,
    _StageRebindingAdapter,
    _surface_call,
    _tool,
    _variant,
)


def test_identity_and_json_text_fields_require_builtin_strings() -> None:
    class RangeMaskingInt(int):
        def __ge__(self, other: object) -> bool:
            return True

        def __le__(self, other: object) -> bool:
            return True

    with pytest.raises(ValueError, match="name"):
        SemanticAction(_AlwaysEqualDigest("search"), {})
    with pytest.raises(ValueError, match="name"):
        SurfaceToolSpec(
            _AlwaysEqualDigest("search"),
            "Search documents",
            {"type": "object"},
        )
    with pytest.raises(ValueError, match="description"):
        SurfaceToolSpec(
            "search",
            _AlwaysEqualDigest("Search documents"),
            {"type": "object"},
        )
    with pytest.raises(ValueError, match="variant_id"):
        SchemaVariant(
            _AlwaysEqualDigest("identity-v1"),
            [_tool()],
            {"operator": "identity"},
        )
    with pytest.raises(ValueError, match=r"unsupported|exact|string"):
        canonical_json_bytes(_AlwaysEqualDigest("value"))
    with pytest.raises(ValueError, match=r"key|string"):
        canonical_json_bytes({_AlwaysEqualDigest("key"): "value"})
    with pytest.raises(ValueError, match=r"unsupported|integer|range"):
        canonical_json_bytes(RangeMaskingInt(2**60))


def test_schema_fingerprint_covers_variant_and_ordered_full_tool_specs() -> None:
    first = SchemaVariant(
        variant_id="identity-v1",
        tools=[_tool("search"), _tool("open")],
        manifest={"seed": 7, "operator": "identity"},
    )
    reordered_mapping_keys = SchemaVariant(
        variant_id="identity-v1",
        tools=[
            SurfaceToolSpec(
                name="search",
                description="Search documents",
                input_schema={
                    "properties": {"query": {"type": "string"}},
                    "type": "object",
                },
            ),
            _tool("open"),
        ],
        manifest={"operator": "identity", "seed": 7},
    )
    reordered_tools = SchemaVariant(
        variant_id="identity-v1",
        tools=[_tool("open"), _tool("search")],
        manifest={"seed": 7, "operator": "identity"},
    )

    assert schema_fingerprint(first) == schema_fingerprint(reordered_mapping_keys)
    assert schema_fingerprint(first) != schema_fingerprint(reordered_tools)


@pytest.mark.parametrize("missing_slot", ["_items", "_index"])
def test_schema_fingerprint_rejects_missing_frozen_mapping_state_without_callbacks(
    missing_slot: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = _variant()
    manifest = variant.manifest
    mapping_type = type(manifest)
    callback_calls = 0

    def callback(_: object) -> tuple[tuple[str, JSONValue], ...]:
        nonlocal callback_calls
        callback_calls += 1
        raise AssertionError("schema validation called a rebound mapping method")

    monkeypatch.setattr(mapping_type, "_raw_items", callback)
    object.__delattr__(manifest, missing_slot)

    with pytest.raises(ValueError):
        schema_fingerprint(variant)
    assert callback_calls == 0


def test_layer_result_is_immutable_typed_and_passes_only_without_diagnostics() -> None:
    diagnostic = ContractDiagnostic(
        code="schema.mapping_mismatch",
        message="Surface mapping differs from expected actions",
        case_id="schema-case-1",
    )
    failed = LayerContractResult(
        layer="schema",
        fingerprint="a" * 64,
        checks_run=1,
        diagnostics=[diagnostic],
    )
    passed = LayerContractResult(
        layer="schema",
        fingerprint="b" * 64,
        checks_run=1,
        diagnostics=[],
    )

    assert failed.diagnostics == (diagnostic,)
    assert failed.passed is False
    assert passed.passed is True
    with pytest.raises(FrozenInstanceError):
        passed.layer = "trace"  # type: ignore[misc]
    with pytest.raises(ValueError, match="checks_run"):
        LayerContractResult("schema", "a" * 64, True, [])
    with pytest.raises(ValueError, match="fingerprint"):
        LayerContractResult("schema", "A" * 64, 1, [])
    with pytest.raises(ValueError, match="case_id"):
        ContractDiagnostic("schema.invalid", "Static message", "../payload")
    with pytest.raises(ValueError, match="message"):
        ContractDiagnostic("schema.invalid", "line one\nline two", "case-1")


def test_layer_passed_revalidates_object_setattr_mutation() -> None:
    result = LayerContractResult(
        "schema",
        "a" * 64,
        1,
        [ContractDiagnostic("schema.failed", "Schema check failed", "case-1")],
    )
    object.__setattr__(result, "diagnostics", ())

    with pytest.raises(ValueError, match="mutated"):
        _ = result.passed


def test_layer_constructor_revalidates_mutated_diagnostic_payload_safety() -> None:
    diagnostic = ContractDiagnostic(
        "schema.failed",
        "Schema check failed",
        "case-1",
    )
    object.__setattr__(diagnostic, "message", "payload\n/private/path")

    with pytest.raises(ValueError, match="message"):
        LayerContractResult("schema", "a" * 64, 1, [diagnostic])


def test_identity_schema_contract_passes_with_complete_unique_probes() -> None:
    variant = _variant("search", "open")
    probes = [
        SchemaProbe(
            case_id=f"schema-{name}",
            surface_tool_name=name,
            surface_call=_surface_call(name),
            expected_actions=[SemanticAction(name, {"value": "alpha"})],
        )
        for name in ("search", "open")
    ]

    result = check_schema_contract(variant, _IdentityAdapter(variant), probes)

    assert result.layer == "schema"
    assert result.fingerprint == schema_fingerprint(variant)
    assert result.checks_run == 2
    assert result.passed is True


def test_schema_diagnostics_keep_inventory_before_mapping_order() -> None:
    variant = _variant("search", "open")
    probe = SchemaProbe(
        "schema-ordered-diagnostics",
        "search",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
    )

    result = check_schema_contract(variant, _BadSchemaAdapter(variant), [probe])

    assert result.fingerprint == schema_fingerprint(variant)
    assert [item.code for item in result.diagnostics] == [
        "schema.missing_tool",
        "schema.mapping_mismatch",
    ]


def test_schema_revalidates_adapter_binding_after_every_parse() -> None:
    variant = _variant()
    adapter = _StageRebindingAdapter(variant, "parse")
    probe = SchemaProbe(
        "schema-rebinding",
        "search",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
    )

    result = check_schema_contract(variant, adapter, [probe])

    assert "schema.variant_rebound" in {item.code for item in result.diagnostics}


def test_schema_contract_rejects_bad_and_nondeterministic_mappings() -> None:
    variant = _variant()
    probe = SchemaProbe(
        "schema-search",
        "search",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
    )

    bad = check_schema_contract(variant, _BadSchemaAdapter(variant), [probe])
    nondeterministic = check_schema_contract(variant, _NondeterministicAdapter(variant), [probe])

    assert {item.code for item in bad.diagnostics} >= {"schema.mapping_mismatch"}
    assert {item.code for item in nondeterministic.diagnostics} >= {
        "schema.nondeterministic_mapping"
    }


def test_schema_contract_allows_repeated_action_name_with_distinct_arguments() -> None:
    class RepeatedActionAdapter(_IdentityAdapter):
        def surface_to_semantic(
            self, surface_call: Mapping[str, JSONValue]
        ) -> tuple[SemanticAction, ...]:
            return (
                SemanticAction("search", {"value": "alpha"}),
                SemanticAction("search", {"value": "beta"}),
            )

    variant = _variant()
    probe = SchemaProbe(
        "schema-repeated-action",
        "search",
        _surface_call(),
        [
            SemanticAction("search", {"value": "alpha"}),
            SemanticAction("search", {"value": "beta"}),
        ],
    )

    result = check_schema_contract(variant, RepeatedActionAdapter(variant), [probe])

    assert result.passed is True


def test_schema_contract_rejects_missing_undeclared_mismatched_and_empty_probes() -> None:
    variant = _variant("search", "open")
    undeclared = SchemaProbe(
        "schema-ghost",
        "ghost",
        _surface_call("different"),
        [SemanticAction("ghost", {"value": "alpha"})],
    )

    result = check_schema_contract(variant, _IdentityAdapter(variant), [undeclared])
    empty = check_schema_contract(variant, _IdentityAdapter(variant), [])

    codes = {item.code for item in result.diagnostics}
    assert codes >= {
        "schema.missing_tool",
        "schema.undeclared_tool",
        "schema.call_name_mismatch",
    }
    assert empty.checks_run == 0
    assert {item.code for item in empty.diagnostics} >= {
        "schema.empty_probes",
        "schema.missing_tool",
    }


def test_schema_contract_uses_json_type_sensitive_mapping_equality() -> None:
    class BoolToIntAdapter(_IdentityAdapter):
        def surface_to_semantic(
            self, surface_call: Mapping[str, JSONValue]
        ) -> tuple[SemanticAction, ...]:
            return (SemanticAction("search", {"value": 1}),)

    variant = _variant()
    probe = SchemaProbe(
        "schema-bool",
        "search",
        _surface_call(value=True),
        [SemanticAction("search", {"value": True})],
    )

    result = check_schema_contract(variant, BoolToIntAdapter(variant), [probe])

    assert "schema.mapping_mismatch" in {item.code for item in result.diagnostics}


def test_schema_probe_snapshots_json_and_schema_exceptions_are_payload_free() -> None:
    surface_call = _surface_call()
    action_arguments: dict[str, JSONValue] = {"value": "alpha"}
    probe = SchemaProbe(
        "schema-search",
        "search",
        surface_call,
        [SemanticAction("search", action_arguments)],
    )
    surface_call["name"] = "changed"
    action_arguments["value"] = "changed"

    result = check_schema_contract(_variant(), _ExplodingAdapter(_variant()), [probe])
    diagnostic_text = " ".join(
        f"{item.code} {item.message} {item.case_id}" for item in result.diagnostics
    )

    assert probe.surface_call["name"] == "search"
    assert probe.expected_actions[0].arguments["value"] == "alpha"
    assert "/private/path" not in diagnostic_text
    assert "secret" not in diagnostic_text
    assert "17" not in diagnostic_text


def test_schema_probe_rejects_unsupported_json_and_detects_object_setattr_mutation() -> None:
    with pytest.raises(ValueError, match="surface_call"):
        SchemaProbe("schema-bad", "search", {"value": float("nan")}, [])
    probe = SchemaProbe(
        "schema-search",
        "search",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
    )
    object.__setattr__(probe, "surface_tool_name", "open")

    result = check_schema_contract(_variant(), _IdentityAdapter(_variant()), [probe])

    assert "schema.invalid_probe" in {item.code for item in result.diagnostics}
