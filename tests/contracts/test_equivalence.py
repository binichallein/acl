"""Behavioral-equivalence contract tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

import toolshift.contracts.denotation as denotation_contracts
import toolshift.contracts.schema as schema_contracts
import toolshift.contracts.state as state_contracts
import toolshift.contracts.trace as trace_contracts
from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.denotation import DenotationCase, check_denotation_contract
from toolshift.contracts.schema import (
    ContractDiagnostic,
    ContractSuiteResult,
    LayerContractResult,
    SchemaProbe,
    check_schema_contract,
    contract_suite_fingerprint,
    evaluate_contract_suite,
    require_dataset_admission,
    schema_fingerprint,
)
from toolshift.contracts.state import StateCase, StateEvidence, check_state_contract
from toolshift.contracts.trace import (
    PhysicalCallEffect,
    TraceCase,
    TraceEvidence,
    check_trace_contract,
)
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    canonical_json_bytes,
)


def _tool(name: str = "search") -> SurfaceToolSpec:
    return SurfaceToolSpec(
        name=name,
        description="Search documents",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


def _variant(*names: str) -> SchemaVariant:
    return SchemaVariant(
        variant_id="identity-v1",
        tools=[_tool(name) for name in (names or ("search",))],
        manifest={"operator": "identity", "seed": 7},
    )


def _surface_call(name: str = "search", value: JSONValue = "alpha") -> dict[str, JSONValue]:
    return {"name": name, "arguments": {"value": value}}


class _IdentityAdapter(SemanticAdapter):
    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        name = surface_call.get("name")
        arguments = surface_call.get("arguments")
        if not isinstance(name, str) or name not in {tool.name for tool in self.variant.tools}:
            raise ValueError("unknown surface tool")
        if not isinstance(arguments, Mapping):
            raise ValueError("surface call arguments must be a mapping")
        return (SemanticAction(name, arguments),)

    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        return ({"name": action.name, "arguments": action.arguments},)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        if len(actions) == 0 and len(base_observation_groups) == 0:
            return {"status": "noop"}
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        return trace.semantic_actions


class _BadSchemaAdapter(_IdentityAdapter):
    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        actions = super().surface_to_semantic(surface_call)
        return (SemanticAction("wrong_action", actions[0].arguments),)


class _NondeterministicAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._calls = 0

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self._calls += 1
        actions = super().surface_to_semantic(surface_call)
        if self._calls % 2 == 0:
            return (SemanticAction(actions[0].name, {"value": "changed"}),)
        return actions


class _ExplodingAdapter(_IdentityAdapter):
    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        raise RuntimeError("payload at /private/path: {'secret': 17}")


class _EmptyCompileAdapter(_IdentityAdapter):
    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        return ()


class _NondeterministicCompileAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._compile_calls = 0

    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self._compile_calls += 1
        suffix: JSONValue = self._compile_calls
        return ({"name": action.name, "arguments": {"value": suffix}},)


class _TwoCallAdapter(_IdentityAdapter):
    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        return (
            {"name": action.name, "arguments": action.arguments},
            {"name": "audit", "arguments": {}},
        )


class _BadObservationAdapter(_IdentityAdapter):
    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        return {"ok": 1}


class _NoOpAdapter(_IdentityAdapter):
    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        return ()


class _ExplodingDenotationAdapter(_IdentityAdapter):
    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        raise RuntimeError("base payload /private/call {'datum': 91}")

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        raise RuntimeError("observation payload /private/observation 92")


class _ActualPipelineAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self.parsed_actions: list[SemanticAction] = []
        self.compiled_actions: list[SemanticAction] = []
        self.wrapped_actions: list[tuple[SemanticAction, ...]] = []

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        action = SemanticAction("search", {"value": "alpha"})
        self.parsed_actions.append(action)
        return (action,)

    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self.compiled_actions.append(action)
        value = "changed" if action is self.parsed_actions[1] else "alpha"
        return (_surface_call(value=value),)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self.wrapped_actions.append(actions)
        if actions and actions[0] is self.parsed_actions[1]:
            return {"items": [2]}
        return {"items": [1]}


class _OneParseFailureAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self.parse_calls = 0
        self.compiled_actions: list[SemanticAction] = []
        self.wrapped_actions: list[tuple[SemanticAction, ...]] = []

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self.parse_calls += 1
        if self.parse_calls == 1:
            raise RuntimeError("parser payload /private/parse 93")
        return (SemanticAction("search", {"value": "alpha"}),)

    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self.compiled_actions.append(action)
        return (_surface_call(),)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self.wrapped_actions.append(actions)
        return {"items": [1]}


class _SecondCompileDriftAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self.compile_counts: dict[int, int] = {}

    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        action_id = id(action)
        count = self.compile_counts.get(action_id, 0) + 1
        self.compile_counts[action_id] = count
        value = "changed" if count == 2 else "alpha"
        return (_surface_call(value=value),)


class _StageRebindingAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant, stage: str) -> None:
        super().__init__(variant)
        self._stage = stage
        self._rebound_variant = SchemaVariant(
            "rebound-v1",
            variant.tools,
            {"operator": "rebound", "seed": 8},
        )

    def _maybe_rebind(self, stage: str) -> None:
        if self._stage == stage:
            self._variant = self._rebound_variant

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self._maybe_rebind("parse")
        name = surface_call["name"]
        arguments = surface_call["arguments"]
        assert isinstance(name, str)
        assert isinstance(arguments, Mapping)
        return (SemanticAction(name, arguments),)

    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self._maybe_rebind("compile")
        return ({"name": action.name, "arguments": action.arguments},)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self._maybe_rebind("wrap")
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self._maybe_rebind("canonicalize")
        return trace.semantic_actions


class _InPlaceVariantMutationAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._mutated = False

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        if not self._mutated:
            replacement = SchemaVariant(
                self._variant.variant_id,
                self._variant.tools,
                {"operator": "mutated", "seed": 9},
            )
            object.__setattr__(self._variant, "manifest", replacement.manifest)
            object.__setattr__(
                self._variant,
                "_manifest_canonical",
                replacement._manifest_canonical,
            )
            self._mutated = True
        return super().surface_to_semantic(surface_call)


class _TransientStageRebindingAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant, stage: str) -> None:
        super().__init__(variant)
        self._original_variant = variant
        self._rebound_variant = SchemaVariant(
            "transient-v1",
            variant.tools,
            {"operator": "transient", "seed": 10},
        )
        self._stage = stage

    def _toggle_binding(self, stage: str) -> None:
        if self._stage == stage:
            self._variant = (
                self._rebound_variant
                if self._variant is self._original_variant
                else self._original_variant
            )

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self._toggle_binding("parse")
        name = surface_call["name"]
        arguments = surface_call["arguments"]
        assert isinstance(name, str)
        assert isinstance(arguments, Mapping)
        return (SemanticAction(name, arguments),)

    def semantic_to_base_calls(
        self, action: SemanticAction
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self._toggle_binding("compile")
        return ({"name": action.name, "arguments": action.arguments},)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self._toggle_binding("wrap")
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self._toggle_binding("canonicalize")
        return trace.semantic_actions


class _DenotationListSwapAdapter(_IdentityAdapter):
    def __init__(
        self,
        variant: SchemaVariant,
        mutable_steps: list[DenotationCase],
        replacement: DenotationCase,
    ) -> None:
        super().__init__(variant)
        self._mutable_steps = mutable_steps
        self._replacement = replacement
        self._parse_calls = 0

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self._parse_calls += 1
        if self._parse_calls == 3:
            self._mutable_steps[0] = self._replacement
        return super().surface_to_semantic(surface_call)


class _AlwaysEqualDigest(str):
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


def _replace_denotation_case_content(
    target: DenotationCase,
    replacement: DenotationCase,
) -> None:
    for name in (
        "case_id",
        "surface_call",
        "expected_actions",
        "expected_base_call_groups",
        "base_observation_groups",
        "expected_surface_observation",
        "_snapshot_fingerprint",
    ):
        object.__setattr__(target, name, getattr(replacement, name))


def _replace_state_case_content(target: StateCase, replacement: StateCase) -> None:
    for name in ("case_id", "episode_id", "_snapshot_fingerprint"):
        object.__setattr__(target, name, getattr(replacement, name))


def _replace_trace_case_content(target: TraceCase, replacement: TraceCase) -> None:
    for name in ("case_id", "episode_id", "_snapshot_fingerprint"):
        object.__setattr__(target, name, getattr(replacement, name))


def _replace_variant_content(
    target: SchemaVariant,
    replacement: SchemaVariant,
) -> None:
    for name in (
        "variant_id",
        "tools",
        "manifest",
        "_manifest_canonical",
    ):
        object.__setattr__(target, name, getattr(replacement, name))


class _LayerBoundaryVariantAdapter(_IdentityAdapter):
    def __init__(
        self,
        original: SchemaVariant,
        intermediate: SchemaVariant,
    ) -> None:
        super().__init__(original)
        self._original_variant = original
        self._intermediate_variant = intermediate
        self._serve_intermediate = False
        self._intermediate_reads = 0

    def begin_intermediate_window(self) -> None:
        self._serve_intermediate = True
        self._intermediate_reads = 0

    @property
    def variant(self) -> SchemaVariant:
        if self._serve_intermediate:
            self._intermediate_reads += 1
            if self._intermediate_reads <= 3:
                return self._intermediate_variant
        return self._original_variant


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
    nondeterministic = check_schema_contract(
        variant, _NondeterministicAdapter(variant), [probe]
    )

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


def test_identity_single_step_denotation_contract_passes() -> None:
    variant = _variant()
    case = DenotationCase(
        case_id="denotation-search",
        surface_call=_surface_call(),
        expected_actions=[SemanticAction("search", {"value": "alpha"})],
        expected_base_call_groups=[[_surface_call()]],
        base_observation_groups=[[{"items": [1, 2]}]],
        expected_surface_observation={"items": [1, 2]},
    )

    result = check_denotation_contract(_IdentityAdapter(variant), [case])

    assert result.layer == "denotation"
    assert result.checks_run == 1
    assert result.passed is True


def test_denotation_runs_both_actual_parse_objects_through_compile_and_wrap() -> None:
    adapter = _ActualPipelineAdapter(_variant())
    case = DenotationCase(
        "denotation-actual-pipeline",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
        [[_surface_call()]],
        [[{"items": [1]}]],
        {"items": [1]},
    )

    result = check_denotation_contract(adapter, [case])

    assert len(adapter.parsed_actions) == 2
    assert len(adapter.compiled_actions) == 4
    assert adapter.compiled_actions[0] is adapter.parsed_actions[0]
    assert adapter.compiled_actions[1] is adapter.parsed_actions[0]
    assert adapter.compiled_actions[2] is adapter.parsed_actions[1]
    assert adapter.compiled_actions[3] is adapter.parsed_actions[1]
    assert len(adapter.wrapped_actions) == 2
    assert adapter.wrapped_actions[0][0] is adapter.parsed_actions[0]
    assert adapter.wrapped_actions[1][0] is adapter.parsed_actions[1]
    assert {item.code for item in result.diagnostics} >= {
        "denotation.nondeterministic_compile",
        "denotation.nondeterministic_observation",
    }


def test_denotation_does_not_fallback_to_expected_actions_after_parse_failure() -> None:
    adapter = _OneParseFailureAdapter(_variant())
    case = DenotationCase(
        "denotation-parse-failure",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
        [[_surface_call()]],
        [[{"items": [1]}]],
        {"items": [1]},
    )

    result = check_denotation_contract(adapter, [case])

    assert "denotation.parse_exception" in {item.code for item in result.diagnostics}
    assert adapter.compiled_actions == []
    assert adapter.wrapped_actions == []


def test_denotation_compiles_each_actual_action_twice_for_determinism() -> None:
    adapter = _SecondCompileDriftAdapter(_variant())
    case = DenotationCase(
        "denotation-same-object-compile-drift",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
        [[_surface_call()]],
        [[{"items": [1]}]],
        {"items": [1]},
    )

    result = check_denotation_contract(adapter, [case])

    assert sorted(adapter.compile_counts.values()) == [2, 2]
    assert "denotation.nondeterministic_compile" in {
        item.code for item in result.diagnostics
    }


def test_denotation_rejects_nondeterministic_parse_compile_and_empty_base_group() -> None:
    variant = _variant()
    case = DenotationCase(
        "denotation-search",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
        [[_surface_call()]],
        [[{"items": []}]],
        {"items": []},
    )

    parse_result = check_denotation_contract(_NondeterministicAdapter(variant), [case])
    compile_result = check_denotation_contract(
        _NondeterministicCompileAdapter(variant), [case]
    )
    empty_result = check_denotation_contract(_EmptyCompileAdapter(variant), [case])

    assert "denotation.nondeterministic_parse" in {
        item.code for item in parse_result.diagnostics
    }
    assert "denotation.nondeterministic_compile" in {
        item.code for item in compile_result.diagnostics
    }
    assert "denotation.compile_exception" in {
        item.code for item in empty_result.diagnostics
    }


def test_denotation_rejects_base_call_and_observation_json_type_drift() -> None:
    class BoolToIntCompileAdapter(_IdentityAdapter):
        def semantic_to_base_calls(
            self, action: SemanticAction
        ) -> tuple[Mapping[str, JSONValue], ...]:
            return ({"name": action.name, "arguments": {"value": 1}},)

    variant = _variant()
    bool_case = DenotationCase(
        "denotation-bool",
        _surface_call(value=True),
        [SemanticAction("search", {"value": True})],
        [[_surface_call(value=True)]],
        [[{"ok": True}]],
        {"ok": True},
    )

    compile_result = check_denotation_contract(BoolToIntCompileAdapter(variant), [bool_case])
    observation_result = check_denotation_contract(_BadObservationAdapter(variant), [bool_case])

    assert "denotation.compile_mismatch" in {
        item.code for item in compile_result.diagnostics
    }
    assert "denotation.observation_mismatch" in {
        item.code for item in observation_result.diagnostics
    }


def test_denotation_rejects_observation_cardinality_mismatch() -> None:
    variant = _variant()
    case = DenotationCase(
        "denotation-two-calls",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
        [[_surface_call(), {"name": "audit", "arguments": {}}]],
        [[{"only": "one"}]],
        {"only": "one"},
    )

    result = check_denotation_contract(_TwoCallAdapter(variant), [case])

    assert "denotation.observation_cardinality" in {
        item.code for item in result.diagnostics
    }


def test_denotation_allows_zero_actions_only_with_zero_groups_and_wraps_each_run() -> None:
    variant = _variant()
    valid = DenotationCase(
        "denotation-noop",
        _surface_call(),
        [],
        [],
        [],
        {"status": "noop"},
    )
    invalid = DenotationCase(
        "denotation-noop-invalid",
        _surface_call(),
        [],
        [[]],
        [[]],
        {"status": "noop"},
    )

    passed = check_denotation_contract(_NoOpAdapter(variant), [valid])
    failed = check_denotation_contract(_NoOpAdapter(variant), [invalid])

    assert passed.passed is True
    assert {item.code for item in failed.diagnostics} >= {
        "denotation.base_group_count",
        "denotation.observation_group_count",
    }


def test_denotation_empty_suite_and_exceptions_fail_without_payload_leakage() -> None:
    variant = _variant()
    case = DenotationCase(
        "denotation-secret",
        _surface_call(),
        [SemanticAction("search", {"value": "alpha"})],
        [[_surface_call()]],
        [[{"secret": "payload"}]],
        {"secret": "payload"},
    )

    empty = check_denotation_contract(_IdentityAdapter(variant), [])
    result = check_denotation_contract(_ExplodingDenotationAdapter(variant), [case])
    diagnostic_text = " ".join(
        f"{item.code} {item.message} {item.case_id}" for item in result.diagnostics
    )

    assert empty.checks_run == 0
    assert "denotation.empty_cases" in {item.code for item in empty.diagnostics}
    assert {item.code for item in result.diagnostics} >= {
        "denotation.compile_exception",
        "denotation.observation_exception",
    }
    assert "/private" not in diagnostic_text
    assert "datum" not in diagnostic_text
    assert "91" not in diagnostic_text
    assert "92" not in diagnostic_text


def test_denotation_case_deep_snapshots_and_rejects_malformed_or_mutated_fields() -> None:
    call = _surface_call()
    base_call = _surface_call()
    observation: dict[str, JSONValue] = {"items": [1]}
    case = DenotationCase(
        "denotation-snapshot",
        call,
        [SemanticAction("search", {"value": "alpha"})],
        [[base_call]],
        [[observation]],
        observation,
    )
    call["name"] = "changed"
    base_call["name"] = "changed"
    cast_items = observation["items"]
    assert isinstance(cast_items, list)
    cast_items.append(2)

    assert case.surface_call["name"] == "search"
    assert case.expected_base_call_groups[0][0]["name"] == "search"
    assert case.base_observation_groups[0][0]["items"] == (1,)
    assert case.expected_surface_observation["items"] == (1,)
    with pytest.raises(ValueError, match="expected_surface_observation"):
        DenotationCase("denotation-bad", {}, [], [], [], float("inf"))

    object.__setattr__(case, "case_id", "denotation-mutated")
    result = check_denotation_contract(_IdentityAdapter(_variant()), [case])
    assert "denotation.invalid_case" in {item.code for item in result.diagnostics}


def test_identity_state_contract_passes_with_opaque_provider_evidence() -> None:
    case = StateCase(case_id="state-episode-1", episode_id="episode-1")

    def provider(requested: StateCase) -> StateEvidence:
        assert requested is case
        return StateEvidence(
            reference_initial_sha256="a" * 64,
            candidate_initial_sha256="a" * 64,
            reference_final_sha256="b" * 64,
            candidate_final_sha256="b" * 64,
            reference_post_reset_sha256="a" * 64,
            candidate_post_reset_sha256="a" * 64,
            reference_collateral_digest="collateral-clean",
            candidate_collateral_digest="collateral-clean",
        )

    result = check_state_contract([case], provider)

    assert result.layer == "state"
    assert result.checks_run == 1
    assert result.passed is True


def _state_evidence(**overrides: str) -> StateEvidence:
    values = {
        "reference_initial_sha256": "a" * 64,
        "candidate_initial_sha256": "a" * 64,
        "reference_final_sha256": "b" * 64,
        "candidate_final_sha256": "b" * 64,
        "reference_post_reset_sha256": "a" * 64,
        "candidate_post_reset_sha256": "a" * 64,
        "reference_collateral_digest": "collateral-clean",
        "candidate_collateral_digest": "collateral-clean",
    }
    values.update(overrides)
    return StateEvidence(**values)


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"candidate_initial_sha256": "c" * 64}, "state.initial_mismatch"),
        ({"candidate_final_sha256": "c" * 64}, "state.final_mismatch"),
        (
            {"candidate_collateral_digest": "collateral-extra"},
            "state.collateral_mismatch",
        ),
        ({"reference_post_reset_sha256": "c" * 64}, "state.reference_reset_mismatch"),
        ({"candidate_post_reset_sha256": "c" * 64}, "state.candidate_reset_mismatch"),
    ],
)
def test_state_contract_rejects_each_state_reset_and_collateral_mismatch(
    overrides: dict[str, str], expected_code: str
) -> None:
    case = StateCase("state-mismatch", "episode-mismatch")

    result = check_state_contract([case], lambda _: _state_evidence(**overrides))

    assert expected_code in {item.code for item in result.diagnostics}


def test_state_contract_empty_cases_and_provider_exceptions_are_payload_free() -> None:
    calls: list[str] = []
    cases = [
        StateCase("state-first", "episode-first"),
        StateCase("state-second", "episode-second"),
    ]

    def provider(case: StateCase) -> StateEvidence:
        calls.append(case.case_id)
        if case.case_id == "state-first":
            raise RuntimeError("database payload /private/state {'row': 404}")
        return _state_evidence()

    empty = check_state_contract([], provider)
    result = check_state_contract(cases, provider)
    diagnostic_text = " ".join(
        f"{item.code} {item.message} {item.case_id}" for item in result.diagnostics
    )

    assert empty.checks_run == 0
    assert "state.empty_cases" in {item.code for item in empty.diagnostics}
    assert calls == ["state-first", "state-second"]
    assert "state.provider_exception" in {item.code for item in result.diagnostics}
    assert "/private" not in diagnostic_text
    assert "row" not in diagnostic_text
    assert "404" not in diagnostic_text


def test_state_contract_rejects_malformed_and_mutated_evidence_and_cases() -> None:
    with pytest.raises(ValueError, match="episode_id"):
        StateCase("state-bad", "../episode")
    with pytest.raises(ValueError, match="reference_initial_sha256"):
        _state_evidence(reference_initial_sha256="A" * 64)

    case = StateCase("state-mutation", "episode-mutation")
    evidence = _state_evidence()
    object.__setattr__(evidence, "candidate_final_sha256", "c" * 64)
    evidence_result = check_state_contract([case], lambda _: evidence)
    object.__setattr__(case, "episode_id", "episode-changed")
    case_result = check_state_contract([case], lambda _: _state_evidence())

    assert "state.provider_exception" in {
        item.code for item in evidence_result.diagnostics
    }
    assert "state.invalid_case" in {item.code for item in case_result.diagnostics}


def test_identity_trace_contract_reconstructs_verified_denotation_steps() -> None:
    variant = _variant()
    action = SemanticAction("search", {"value": "alpha"})
    base_call = _surface_call()
    step = DenotationCase(
        "denotation-trace-step",
        _surface_call(),
        [action],
        [[base_call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    reference_trace = ExecutionTrace([_surface_call()], [action], [base_call])
    candidate_trace = ExecutionTrace([_surface_call()], [action], [base_call])
    call_digest = hashlib.sha256(canonical_json_bytes(base_call)).hexdigest()
    effect = PhysicalCallEffect(
        call_index=0,
        base_call_fingerprint=call_digest,
        effect_digest="effect-clean",
    )
    case = TraceCase("trace-episode-1", "episode-1")

    def provider(requested: TraceCase) -> TraceEvidence:
        assert requested is case
        return TraceEvidence(
            reference_trace=reference_trace,
            candidate_trace=candidate_trace,
            candidate_steps=[step],
            reference_score={"success": True, "score": 1.0},
            candidate_score={"success": True, "score": 1.0},
            reference_effects=[effect],
            candidate_effects=[effect],
        )

    result = check_trace_contract(_IdentityAdapter(variant), [case], provider, [step])

    assert result.layer == "trace"
    assert result.checks_run == 1
    assert result.passed is True


def test_trace_requires_a_registered_surface_step_but_allows_zero_actions() -> None:
    variant = _variant()
    adapter = _NoOpAdapter(variant)
    call = _surface_call()
    probe = SchemaProbe("schema-zero-step", "search", call, [])
    step = DenotationCase(
        "denotation-zero-step",
        call,
        [],
        [],
        [],
        {"status": "noop"},
    )
    state_case = StateCase("state-zero-step", "episode-zero-step")
    trace_case = TraceCase("trace-zero-step", "episode-zero-step")
    empty_trace = ExecutionTrace([], [], [])
    empty_evidence = TraceEvidence(
        empty_trace,
        empty_trace,
        [],
        1,
        1,
        [],
        [],
    )

    empty_suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        lambda _: empty_evidence,
    )

    assert empty_suite.results[3].checks_run == 0
    assert "trace.empty_steps" in {
        item.code for item in empty_suite.results[3].diagnostics
    }
    with pytest.raises(ValueError, match=r"non-vacuous|pass|provenance"):
        require_dataset_admission(variant, empty_suite)

    zero_action_trace = ExecutionTrace([call], [], [])
    zero_action_evidence = TraceEvidence(
        zero_action_trace,
        zero_action_trace,
        [step],
        1,
        1,
        [],
        [],
    )
    zero_action_suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        lambda _: zero_action_evidence,
    )

    assert zero_action_suite.results[3].checks_run == 1
    assert require_dataset_admission(variant, zero_action_suite) is zero_action_suite


def _effect(
    call: Mapping[str, JSONValue], index: int = 0, digest: str = "effect-clean"
) -> PhysicalCallEffect:
    return PhysicalCallEffect(
        index,
        hashlib.sha256(canonical_json_bytes(call)).hexdigest(),
        digest,
    )


def _single_step_trace_fixture() -> tuple[
    DenotationCase, TraceCase, ExecutionTrace, PhysicalCallEffect
]:
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    step = DenotationCase(
        "denotation-trace-step",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    trace = ExecutionTrace([call], [action], [call])
    return step, TraceCase("trace-case", "episode-trace"), trace, _effect(call)


def test_trace_checker_never_reruns_per_step_adapter_mappings() -> None:
    class TraceOnlyAdapter(_IdentityAdapter):
        def surface_to_semantic(
            self, surface_call: Mapping[str, JSONValue]
        ) -> tuple[SemanticAction, ...]:
            raise AssertionError("surface mapping must not run")

        def semantic_to_base_calls(
            self, action: SemanticAction
        ) -> tuple[Mapping[str, JSONValue], ...]:
            raise AssertionError("base compilation must not run")

        def base_observation_to_surface(
            self,
            surface_call: Mapping[str, JSONValue],
            actions: tuple[SemanticAction, ...],
            base_observation_groups: tuple[tuple[JSONValue, ...], ...],
        ) -> JSONValue:
            raise AssertionError("observation wrapping must not run")

    step, case, trace, effect = _single_step_trace_fixture()
    evidence = TraceEvidence(
        trace,
        trace,
        [step],
        {"score": 1},
        {"score": 1},
        [effect],
        [effect],
    )

    result = check_trace_contract(
        TraceOnlyAdapter(_variant()), [case], lambda _: evidence, [step]
    )

    assert result.passed is True


def test_trace_rejects_reordered_or_changed_native_base_calls() -> None:
    first = _surface_call()
    second = {"name": "audit", "arguments": {}}
    action = SemanticAction("search", {"value": "alpha"})
    step = DenotationCase(
        "denotation-two-physical",
        first,
        [action],
        [[first, second]],
        [[{"items": []}, {"audit": True}]],
        {"items": []},
    )
    reference = ExecutionTrace([first], [action], [first, second])
    candidate = ExecutionTrace([first], [action], [second, first])
    reference_effects = [_effect(first, 0), _effect(second, 1)]
    candidate_effects = [_effect(second, 0), _effect(first, 1)]
    evidence = TraceEvidence(
        reference,
        candidate,
        [step],
        1,
        1,
        reference_effects,
        candidate_effects,
    )

    result = check_trace_contract(
        _IdentityAdapter(_variant()),
        [TraceCase("trace-reordered", "episode-reordered")],
        lambda _: evidence,
        [step],
    )

    assert {item.code for item in result.diagnostics} >= {
        "trace.base_reconstruction_mismatch",
        "trace.base_call_mismatch",
    }


def test_trace_rejects_semantic_reordering_and_surface_reconstruction_drift() -> None:
    search = SemanticAction("search", {"value": "alpha"})
    opened = SemanticAction("open", {"value": "alpha"})
    search_call = _surface_call("search")
    open_call = _surface_call("open")
    step = DenotationCase(
        "denotation-merged",
        {"name": "merged", "arguments": {"value": "alpha"}},
        [search, opened],
        [[search_call], [open_call]],
        [[{"search": True}], [{"open": True}]],
        {"ok": True},
    )
    reference = ExecutionTrace(
        [{"name": "reference", "arguments": {}}],
        [opened, search],
        [search_call, open_call],
    )
    candidate = ExecutionTrace(
        [{"name": "lookalike", "arguments": {"value": "alpha"}}],
        [search, opened],
        [search_call, open_call],
    )
    effects = [_effect(search_call, 0), _effect(open_call, 1)]
    evidence = TraceEvidence(reference, candidate, [step], 1, 1, effects, effects)

    result = check_trace_contract(
        _IdentityAdapter(_variant("search", "open")),
        [TraceCase("trace-semantic", "episode-semantic")],
        lambda _: evidence,
        [step],
    )

    assert {item.code for item in result.diagnostics} >= {
        "trace.surface_reconstruction_mismatch",
        "trace.semantic_mismatch",
    }


def test_trace_rejects_constant_canonicalizer_unbound_to_reconstructed_semantics() -> None:
    class ConstantCanonicalAdapter(_IdentityAdapter):
        def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
            return ()

    step, case, candidate_trace, effect = _single_step_trace_fixture()
    reference_trace = ExecutionTrace(
        candidate_trace.surface_calls,
        [SemanticAction("delete_all", {})],
        candidate_trace.base_calls,
    )
    evidence = TraceEvidence(
        reference_trace,
        candidate_trace,
        [step],
        1,
        1,
        [effect],
        [effect],
    )

    result = check_trace_contract(
        ConstantCanonicalAdapter(_variant()), [case], lambda _: evidence, [step]
    )

    assert "trace.canonical_semantic_channel_mismatch" in {
        item.code for item in result.diagnostics
    }


def test_trace_rejects_effect_score_and_alignment_mismatches_type_sensitively() -> None:
    step, case, trace, effect = _single_step_trace_fixture()
    bad_candidate_effect = PhysicalCallEffect(
        0, effect.base_call_fingerprint, "effect-extra"
    )
    evidence = TraceEvidence(
        trace,
        trace,
        [step],
        {"score": True},
        {"score": 1},
        [effect],
        [bad_candidate_effect],
    )

    result = check_trace_contract(
        _IdentityAdapter(_variant()), [case], lambda _: evidence, [step]
    )

    assert {item.code for item in result.diagnostics} >= {
        "trace.effect_mismatch",
        "trace.score_mismatch",
    }

    bad_alignment = TraceEvidence(
        trace,
        trace,
        [step],
        1,
        1,
        [effect],
        [PhysicalCallEffect(1, effect.base_call_fingerprint, "effect-clean")],
    )
    alignment_result = check_trace_contract(
        _IdentityAdapter(_variant()), [case], lambda _: bad_alignment, [step]
    )
    assert "trace.candidate_effect_alignment" in {
        item.code for item in alignment_result.diagnostics
    }


def test_trace_rejects_unverified_lookalike_denotation_step_by_identity() -> None:
    step, case, trace, effect = _single_step_trace_fixture()
    lookalike = DenotationCase(
        step.case_id,
        step.surface_call,
        step.expected_actions,
        step.expected_base_call_groups,
        step.base_observation_groups,
        step.expected_surface_observation,
    )
    evidence = TraceEvidence(trace, trace, [lookalike], 1, 1, [effect], [effect])

    result = check_trace_contract(
        _IdentityAdapter(_variant()), [case], lambda _: evidence, [step]
    )

    assert "trace.unverified_step" in {item.code for item in result.diagnostics}


def test_trace_empty_cases_and_provider_exceptions_are_payload_free() -> None:
    step, _, _, _ = _single_step_trace_fixture()
    cases = [
        TraceCase("trace-first", "episode-first"),
        TraceCase("trace-second", "episode-second"),
    ]
    calls: list[str] = []

    def provider(case: TraceCase) -> TraceEvidence:
        calls.append(case.case_id)
        raise RuntimeError("trace payload /private/trace {'score': 707}")

    empty = check_trace_contract(_IdentityAdapter(_variant()), [], provider, [step])
    result = check_trace_contract(_IdentityAdapter(_variant()), cases, provider, [step])
    diagnostic_text = " ".join(
        f"{item.code} {item.message} {item.case_id}" for item in result.diagnostics
    )

    assert empty.checks_run == 0
    assert "trace.empty_cases" in {item.code for item in empty.diagnostics}
    assert calls == ["trace-first", "trace-second"]
    assert "trace.provider_exception" in {item.code for item in result.diagnostics}
    assert "/private" not in diagnostic_text
    assert "707" not in diagnostic_text


def test_trace_evidence_deep_snapshots_and_rejects_malformed_or_mutated_values() -> None:
    step, case, trace, effect = _single_step_trace_fixture()
    score: dict[str, JSONValue] = {"parts": [1]}
    evidence = TraceEvidence(trace, trace, [step], score, score, [effect], [effect])
    score_parts = score["parts"]
    assert isinstance(score_parts, list)
    score_parts.append(2)

    assert evidence.reference_score["parts"] == (1,)
    with pytest.raises(ValueError, match="call_index"):
        PhysicalCallEffect(True, effect.base_call_fingerprint, "effect-clean")
    with pytest.raises(ValueError, match="reference_score"):
        TraceEvidence(trace, trace, [step], float("nan"), 1, [effect], [effect])

    object.__setattr__(evidence.reference_effects[0], "call_index", 9)
    evidence_result = check_trace_contract(
        _IdentityAdapter(_variant()), [case], lambda _: evidence, [step]
    )
    object.__setattr__(case, "episode_id", "episode-mutated")
    case_result = check_trace_contract(
        _IdentityAdapter(_variant()), [case], lambda _: evidence, [step]
    )

    assert "trace.provider_exception" in {
        item.code for item in evidence_result.diagnostics
    }
    assert "trace.invalid_case" in {item.code for item in case_result.diagnostics}


def test_identity_contract_suite_passes_all_layers_and_hard_admission() -> None:
    variant = _variant()
    adapter = _IdentityAdapter(variant)
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    schema_probe = SchemaProbe("schema-suite", "search", call, [action])
    denotation_case = DenotationCase(
        "denotation-suite",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    state_case = StateCase("state-suite", "episode-suite")
    trace_case = TraceCase("trace-suite", "episode-suite")
    trace = ExecutionTrace([call], [action], [call])
    effect = _effect(call)
    trace_evidence = TraceEvidence(
        trace,
        trace,
        [denotation_case],
        {"success": True},
        {"success": True},
        [effect],
        [effect],
    )

    suite = evaluate_contract_suite(
        variant=variant,
        adapter=adapter,
        schema_probes=[schema_probe],
        denotation_cases=[denotation_case],
        state_cases=[state_case],
        state_provider=lambda _: _state_evidence(),
        trace_cases=[trace_case],
        trace_provider=lambda _: trace_evidence,
    )

    assert isinstance(suite, ContractSuiteResult)
    assert [result.layer for result in suite.results] == [
        "schema",
        "denotation",
        "state",
        "trace",
    ]
    assert suite.schema_fingerprint == schema_fingerprint(variant)
    assert suite.passed is True
    assert contract_suite_fingerprint(suite.schema_fingerprint, suite.results) == (
        suite.fingerprint
    )
    assert require_dataset_admission(variant, suite) is suite


def _identity_suite_components(
    *,
    state_episode: str = "episode-suite",
    trace_episode: str = "episode-suite",
) -> tuple[
    SchemaVariant,
    _IdentityAdapter,
    SchemaProbe,
    DenotationCase,
    StateCase,
    TraceCase,
    TraceEvidence,
]:
    variant = _variant()
    adapter = _IdentityAdapter(variant)
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    probe = SchemaProbe("schema-suite-helper", "search", call, [action])
    step = DenotationCase(
        "denotation-suite-helper",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    state_case = StateCase("state-suite-helper", state_episode)
    trace_case = TraceCase("trace-suite-helper", trace_episode)
    trace = ExecutionTrace([call], [action], [call])
    effect = _effect(call)
    evidence = TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])
    return variant, adapter, probe, step, state_case, trace_case, evidence


def _evaluate_identity_components(
    *,
    state_episode: str = "episode-suite",
    trace_episode: str = "episode-suite",
    adapter_override: SemanticAdapter | None = None,
    state_provider_override: object | None = None,
    trace_evidence_override: TraceEvidence | None = None,
) -> tuple[SchemaVariant, ContractSuiteResult]:
    variant, adapter, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components(
            state_episode=state_episode,
            trace_episode=trace_episode,
        )
    )
    selected_adapter = adapter if adapter_override is None else adapter_override
    selected_state_provider = (
        (lambda _: _state_evidence())
        if state_provider_override is None
        else state_provider_override
    )
    selected_trace_evidence = (
        trace_evidence if trace_evidence_override is None else trace_evidence_override
    )
    suite = evaluate_contract_suite(
        variant,
        selected_adapter,
        [probe],
        [step],
        [state_case],
        selected_state_provider,
        [trace_case],
        lambda _: selected_trace_evidence,
    )
    return variant, suite


@pytest.mark.parametrize("stage", ["parse", "compile", "wrap", "canonicalize"])
def test_suite_rejects_adapter_rebinding_during_every_adapter_stage(stage: str) -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    adapter = _StageRebindingAdapter(variant, stage)

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        lambda _: trace_evidence,
    )

    assert "suite.variant_binding_mismatch" in {
        item.code for item in suite.results[0].diagnostics
    }
    assert suite.passed is False
    with pytest.raises(ValueError, match=r"pass|binding"):
        require_dataset_admission(variant, suite)


def test_suite_uses_initial_fingerprint_against_in_place_variant_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    initial_fingerprint = schema_fingerprint(variant)
    adapter = _InPlaceVariantMutationAdapter(variant)
    monkeypatch.setattr(SchemaVariant, "__eq__", lambda self, other: True)

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        lambda _: trace_evidence,
    )

    assert schema_fingerprint(variant) != initial_fingerprint
    assert "suite.variant_binding_mismatch" in {
        item.code for item in suite.results[0].diagnostics
    }
    assert suite.passed is False
    with pytest.raises(ValueError, match=r"fingerprint|pass|binding"):
        require_dataset_admission(variant, suite)


@pytest.mark.parametrize(
    ("stage", "expected_code"),
    [
        ("parse", "schema.variant_rebound"),
        ("compile", "denotation.variant_rebound"),
        ("wrap", "denotation.variant_rebound"),
        ("canonicalize", "trace.variant_rebound"),
    ],
)
def test_suite_rejects_transient_rebinding_at_each_adapter_boundary(
    stage: str,
    expected_code: str,
) -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    suite = evaluate_contract_suite(
        variant,
        _TransientStageRebindingAdapter(variant, stage),
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        lambda _: trace_evidence,
    )

    assert expected_code in {
        diagnostic.code
        for result in suite.results
        for diagnostic in result.diagnostics
    }
    assert suite.passed is False
    with pytest.raises(ValueError, match=r"pass|binding"):
        require_dataset_admission(variant, suite)


def test_suite_freezes_denotation_identity_before_adapter_can_swap_input_list() -> None:
    variant = _variant()
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    checked_step = DenotationCase(
        "denotation-checked-step",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    unchecked_action = SemanticAction("delete_all", {})
    unchecked_call = {"name": "delete_all", "arguments": {}}
    unchecked_step = DenotationCase(
        "denotation-unchecked-step",
        unchecked_call,
        [unchecked_action],
        [[unchecked_call]],
        [[{"deleted": True}]],
        {"deleted": True},
    )
    mutable_steps = [checked_step]
    adapter = _DenotationListSwapAdapter(variant, mutable_steps, unchecked_step)
    trace = ExecutionTrace([unchecked_call], [unchecked_action], [unchecked_call])
    effect = _effect(unchecked_call)
    trace_evidence = TraceEvidence(
        trace,
        trace,
        [unchecked_step],
        1,
        1,
        [effect],
        [effect],
    )

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [SchemaProbe("schema-toctou", "search", call, [action])],
        mutable_steps,
        [StateCase("state-toctou", "episode-toctou")],
        lambda _: _state_evidence(),
        [TraceCase("trace-toctou", "episode-toctou")],
        lambda _: trace_evidence,
    )

    assert mutable_steps[0] is unchecked_step
    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_UNCHECKED_STEP")
    assert "trace.unverified_step" in {
        item.code for item in suite.results[3].diagnostics
    }


def test_suite_rejects_sequence_subclass_before_it_can_change_entry_baseline() -> None:
    variant, adapter, probe, step, state_case, trace_case, _ = (
        _identity_suite_components()
    )
    replacement_call = _surface_call(value="beta")
    replacement = DenotationCase(
        step.case_id,
        replacement_call,
        [SemanticAction("search", {"value": "beta"})],
        [[replacement_call]],
        [[{"items": [2]}]],
        {"items": [2]},
    )

    class MutatingProbeList(list[SchemaProbe]):
        def __iter__(self):
            _replace_denotation_case_content(step, replacement)
            return super().__iter__()

    def trace_provider(_: TraceCase) -> TraceEvidence:
        trace = ExecutionTrace(
            [step.surface_call],
            step.expected_actions,
            step.expected_base_call_groups[0],
        )
        effect = _effect(step.expected_base_call_groups[0][0])
        return TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])

    suite = evaluate_contract_suite(
        variant,
        adapter,
        MutatingProbeList([probe]),
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_SEQUENCE_SUBCLASS_ENTRY_REWRITE")
    assert step.surface_call == _surface_call()


def test_suite_captures_all_entry_digests_before_validating_any_case() -> None:
    variant, adapter, probe, step, state_case, trace_case, _ = (
        _identity_suite_components()
    )
    replacement_call = _surface_call(value="beta")
    replacement = DenotationCase(
        step.case_id,
        replacement_call,
        [SemanticAction("search", {"value": "beta"})],
        [[replacement_call]],
        [[{"items": [2]}]],
        {"items": [2]},
    )

    class MutatingCall(Mapping[str, JSONValue]):
        def __init__(self, values: Mapping[str, JSONValue]) -> None:
            self._values = values

        def __getitem__(self, key: str) -> JSONValue:
            return self._values[key]

        def __iter__(self):
            _replace_denotation_case_content(step, replacement)
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    object.__setattr__(probe, "surface_call", MutatingCall(probe.surface_call))

    def trace_provider(_: TraceCase) -> TraceEvidence:
        trace = ExecutionTrace(
            [step.surface_call],
            step.expected_actions,
            step.expected_base_call_groups[0],
        )
        effect = _effect(step.expected_base_call_groups[0][0])
        return TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_SEQUENTIAL_ENTRY_BASELINE_REWRITE")
    assert "suite.case_snapshot_mismatch" in {
        item.code for result in suite.results for item in result.diagnostics
    }


def test_entry_shape_check_never_invokes_injected_mapping_callback() -> None:
    variant, adapter, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    rebound_variant = SchemaVariant(
        "callback-rebound-v1",
        variant.tools,
        {"operator": "callback-rebound", "seed": 12},
    )

    class CallbackCall(Mapping[str, JSONValue]):
        def __init__(self, values: Mapping[str, JSONValue]) -> None:
            self._values = values
            self.armed = False
            self.fired = False

        def __getitem__(self, key: str) -> JSONValue:
            return self._values[key]

        def __iter__(self):
            if self.armed:
                self.fired = True
                adapter._variant = rebound_variant
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    callback_call = CallbackCall(probe.surface_call)
    object.__setattr__(probe, "surface_call", callback_call)

    def trace_provider(_: TraceCase) -> TraceEvidence:
        callback_call.armed = True
        return trace_evidence

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_CALLBACK_CAPABLE_ENTRY_FIELD")
    assert callback_call.fired is False
    assert suite.passed is False


def test_entry_shape_check_never_invokes_proxied_mapping_callback() -> None:
    variant, adapter, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    rebound_variant = SchemaVariant(
        "proxied-callback-v1",
        variant.tools,
        {"operator": "proxied-callback", "seed": 13},
    )

    class CallbackCall(Mapping[str, JSONValue]):
        def __init__(self, values: Mapping[str, JSONValue]) -> None:
            self._values = values
            self.armed = False
            self.fired = False

        def __getitem__(self, key: str) -> JSONValue:
            return self._values[key]

        def __iter__(self):
            if self.armed:
                self.fired = True
                adapter._variant = rebound_variant
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    callback_call = CallbackCall(probe.surface_call)
    object.__setattr__(probe, "surface_call", MappingProxyType(callback_call))

    def trace_provider(_: TraceCase) -> TraceEvidence:
        callback_call.armed = True
        return trace_evidence

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_PROXIED_CALLBACK_ENTRY_FIELD")
    assert callback_call.fired is False
    assert suite.passed is False


def test_action_shape_check_never_invokes_proxied_mapping_callback() -> None:
    action = SemanticAction("search", {"value": "alpha"})

    class CallbackArguments(Mapping[str, JSONValue]):
        def __init__(self, values: Mapping[str, JSONValue]) -> None:
            self._values = values
            self.fired = False

        def __getitem__(self, key: str) -> JSONValue:
            return self._values[key]

        def __iter__(self):
            self.fired = True
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    callback_arguments = CallbackArguments(action.arguments)
    object.__setattr__(
        action,
        "arguments",
        MappingProxyType(callback_arguments),
    )

    assert schema_contracts._semantic_action_has_canonical_shape(action) is False
    assert callback_arguments.fired is False


def test_suite_freezes_checked_state_episode_before_provider_can_swap_list() -> None:
    variant, adapter, probe, step, _, _, _ = _identity_suite_components()
    checked_state_case = StateCase("state-checked-episode", "episode-checked")
    unchecked_state_case = StateCase("state-unchecked-episode", "episode-unchecked")
    mutable_state_cases = [checked_state_case]
    trace_case = TraceCase("trace-unchecked-episode", "episode-unchecked")
    action = step.expected_actions[0]
    call = step.surface_call
    trace = ExecutionTrace([call], [action], [call])
    effect = _effect(call)
    trace_evidence = TraceEvidence(
        trace,
        trace,
        [step],
        1,
        1,
        [effect],
        [effect],
    )

    def swapping_state_provider(requested: StateCase) -> StateEvidence:
        assert requested is checked_state_case
        mutable_state_cases[0] = unchecked_state_case
        return _state_evidence()

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        mutable_state_cases,
        swapping_state_provider,
        [trace_case],
        lambda _: trace_evidence,
    )

    assert mutable_state_cases[0] is unchecked_state_case
    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_DIFFERENT_CHECKED_EPISODES")
    assert "suite.episode_set_mismatch" in {
        item.code for item in suite.results[3].diagnostics
    }


def test_suite_conversion_exceptions_still_return_all_four_layer_results() -> None:
    class ExplodingList(list[object]):
        def __iter__(self):
            raise RuntimeError("conversion payload /private/sequence 919")

    variant = _variant()
    exploding = ExplodingList([object()])

    suite = evaluate_contract_suite(
        variant,
        _IdentityAdapter(variant),
        exploding,  # type: ignore[arg-type]
        exploding,
        exploding,
        lambda _: _state_evidence(),
        exploding,
        lambda _: None,
    )
    diagnostic_text = " ".join(
        f"{item.code} {item.message} {item.case_id}"
        for result in suite.results
        for item in result.diagnostics
    )

    assert [result.layer for result in suite.results] == [
        "schema",
        "denotation",
        "state",
        "trace",
    ]
    assert all(result.diagnostics for result in suite.results)
    assert "/private" not in diagnostic_text
    assert "919" not in diagnostic_text


def test_nonexact_episode_identifiers_fail_closed_without_exception_text() -> None:
    class ExplodingIdentifier(str):
        def __lt__(self, other: object) -> bool:
            raise RuntimeError("identifier payload /private/episodes 777")

    variant, adapter, probe, step, _, _, trace_evidence = (
        _identity_suite_components()
    )
    state_cases = [
        StateCase("state-id-first", "episode-a"),
        StateCase("state-id-second", "episode-b"),
    ]
    trace_cases = [
        TraceCase("trace-id-first", "episode-a"),
        TraceCase("trace-id-second", "episode-b"),
    ]
    for case in (*state_cases, *trace_cases):
        object.__setattr__(
            case,
            "episode_id",
            ExplodingIdentifier(case.episode_id),
        )

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        state_cases,
        lambda _: _state_evidence(),
        trace_cases,
        lambda _: trace_evidence,
    )

    assert tuple(result.layer for result in suite.results) == (
        "schema",
        "denotation",
        "state",
        "trace",
    )
    assert suite.passed is False
    diagnostic_text = " ".join(
        f"{item.code} {item.message} {item.case_id}"
        for result in suite.results
        for item in result.diagnostics
    )
    assert "/private" not in diagnostic_text
    assert "777" not in diagnostic_text


def test_suite_rejects_denotation_content_changed_between_layers() -> None:
    variant = _variant()
    adapter = _IdentityAdapter(variant)
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    step = DenotationCase(
        "denotation-cross-layer",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    replacement_action = SemanticAction("delete_all", {})
    replacement_call = {"name": "delete_all", "arguments": {}}
    replacement = DenotationCase(
        step.case_id,
        replacement_call,
        [replacement_action],
        [[replacement_call]],
        [[{"deleted": True}]],
        {"deleted": True},
    )

    def state_provider(_: StateCase) -> StateEvidence:
        _replace_denotation_case_content(step, replacement)
        return _state_evidence()

    def trace_provider(_: TraceCase) -> TraceEvidence:
        trace = ExecutionTrace(
            [step.surface_call],
            step.expected_actions,
            step.expected_base_call_groups[0],
        )
        effect = _effect(step.expected_base_call_groups[0][0])
        return TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [SchemaProbe("schema-cross-layer", "search", call, [action])],
        [step],
        [StateCase("state-cross-layer", "episode-cross-layer")],
        state_provider,
        [TraceCase("trace-cross-layer", "episode-cross-layer")],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_CROSS_LAYER_CASE_CONTENT")
    assert "suite.case_snapshot_mismatch" in {
        item.code for item in suite.results[2].diagnostics
    }


def test_sha256_subclass_cannot_mask_changed_entry_content() -> None:
    variant = _variant()
    adapter = _IdentityAdapter(variant)
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    step = DenotationCase(
        "denotation-digest-subclass",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    entry_fingerprint = step._snapshot_fingerprint
    replacement_call = {"name": "delete_all", "arguments": {}}
    replacement = DenotationCase(
        step.case_id,
        replacement_call,
        [SemanticAction("delete_all", {})],
        [[replacement_call]],
        [[{"deleted": True}]],
        {"deleted": True},
    )

    def state_provider(_: StateCase) -> StateEvidence:
        _replace_denotation_case_content(step, replacement)
        object.__setattr__(
            step,
            "_snapshot_fingerprint",
            _AlwaysEqualDigest(entry_fingerprint),
        )
        return _state_evidence()

    def trace_provider(_: TraceCase) -> TraceEvidence:
        trace = ExecutionTrace(
            [step.surface_call],
            step.expected_actions,
            step.expected_base_call_groups[0],
        )
        effect = _effect(step.expected_base_call_groups[0][0])
        return TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [SchemaProbe("schema-digest-subclass", "search", call, [action])],
        [step],
        [StateCase("state-digest-subclass", "episode-digest-subclass")],
        state_provider,
        [TraceCase("trace-digest-subclass", "episode-digest-subclass")],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_NONEXACT_SHA256_VALUE")
    assert "suite.case_snapshot_mismatch" in {
        item.code for item in suite.results[2].diagnostics
    }


def test_direct_checkers_reject_nonexact_private_digest_values() -> None:
    variant = _variant()
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()

    probe = SchemaProbe("schema-nonexact-digest", "search", call, [action])
    object.__setattr__(
        probe,
        "_snapshot_fingerprint",
        _AlwaysEqualDigest(probe._snapshot_fingerprint),
    )
    schema_result = check_schema_contract(variant, _IdentityAdapter(variant), [probe])

    denotation_case = DenotationCase(
        "denotation-nonexact-digest",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    object.__setattr__(
        denotation_case,
        "_snapshot_fingerprint",
        _AlwaysEqualDigest(denotation_case._snapshot_fingerprint),
    )
    denotation_result = check_denotation_contract(
        _IdentityAdapter(variant),
        [denotation_case],
    )

    state_case = StateCase("state-nonexact-digest", "episode-nonexact-digest")
    object.__setattr__(
        state_case,
        "_snapshot_fingerprint",
        _AlwaysEqualDigest(state_case._snapshot_fingerprint),
    )
    state_case_result = check_state_contract([state_case], lambda _: _state_evidence())
    state_evidence = _state_evidence()
    object.__setattr__(
        state_evidence,
        "_snapshot_fingerprint",
        _AlwaysEqualDigest(state_evidence._snapshot_fingerprint),
    )
    state_evidence_result = check_state_contract(
        [StateCase("state-evidence-digest", "episode-evidence-digest")],
        lambda _: state_evidence,
    )

    step, trace_case, trace, effect = _single_step_trace_fixture()
    trace_evidence = TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])
    object.__setattr__(
        trace_case,
        "_snapshot_fingerprint",
        _AlwaysEqualDigest(trace_case._snapshot_fingerprint),
    )
    trace_case_result = check_trace_contract(
        _IdentityAdapter(variant),
        [trace_case],
        lambda _: trace_evidence,
        [step],
    )

    step, trace_case, trace, effect = _single_step_trace_fixture()
    trace_evidence = TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])
    object.__setattr__(
        trace_evidence,
        "_snapshot_fingerprint",
        _AlwaysEqualDigest(trace_evidence._snapshot_fingerprint),
    )
    trace_evidence_result = check_trace_contract(
        _IdentityAdapter(variant),
        [trace_case],
        lambda _: trace_evidence,
        [step],
    )

    step, trace_case, trace, effect = _single_step_trace_fixture()
    trace_evidence = TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])
    mutated_effect = trace_evidence.reference_effects[0]
    object.__setattr__(
        mutated_effect,
        "_snapshot_fingerprint",
        _AlwaysEqualDigest(mutated_effect._snapshot_fingerprint),
    )
    effect_result = check_trace_contract(
        _IdentityAdapter(variant),
        [trace_case],
        lambda _: trace_evidence,
        [step],
    )

    observed_codes = {
        "schema": {item.code for item in schema_result.diagnostics},
        "denotation": {item.code for item in denotation_result.diagnostics},
        "state_case": {item.code for item in state_case_result.diagnostics},
        "state_evidence": {item.code for item in state_evidence_result.diagnostics},
        "trace_case": {item.code for item in trace_case_result.diagnostics},
        "trace_evidence": {item.code for item in trace_evidence_result.diagnostics},
        "effect": {item.code for item in effect_result.diagnostics},
    }
    expected_codes = {
        "schema": "schema.invalid_probe",
        "denotation": "denotation.invalid_case",
        "state_case": "state.invalid_case",
        "state_evidence": "state.provider_exception",
        "trace_case": "trace.invalid_case",
        "trace_evidence": "trace.provider_exception",
        "effect": "trace.provider_exception",
    }
    assert {
        name: expected
        for name, expected in expected_codes.items()
        if expected not in observed_codes[name]
    } == {}


def test_public_digest_and_identifier_subclasses_cannot_mask_mismatches() -> None:
    state_case = StateCase("state-public-text", "episode-public-text")

    def wrong_final_provider(_: StateCase) -> StateEvidence:
        return _state_evidence(
            candidate_final_sha256=_AlwaysEqualDigest("f" * 64),
        )

    def wrong_collateral_provider(_: StateCase) -> StateEvidence:
        return _state_evidence(
            candidate_collateral_digest=_AlwaysEqualDigest("collateral-extra"),
        )

    wrong_final = check_state_contract([state_case], wrong_final_provider)
    wrong_collateral = check_state_contract([state_case], wrong_collateral_provider)

    step, trace_case, trace, reference_effect = _single_step_trace_fixture()

    def wrong_effect_provider(_: TraceCase) -> TraceEvidence:
        candidate_effect = PhysicalCallEffect(
            reference_effect.call_index,
            reference_effect.base_call_fingerprint,
            _AlwaysEqualDigest("effect-extra"),
        )
        return TraceEvidence(
            trace,
            trace,
            [step],
            1,
            1,
            [reference_effect],
            [candidate_effect],
        )

    wrong_effect = check_trace_contract(
        _IdentityAdapter(_variant()),
        [trace_case],
        wrong_effect_provider,
        [step],
    )

    assert wrong_final.passed is False
    assert wrong_collateral.passed is False
    assert wrong_effect.passed is False


def test_physical_call_index_subclass_cannot_mask_alignment_mismatch() -> None:
    class AlwaysEqualIndex(int):
        def __eq__(self, other: object) -> bool:
            return True

        def __ne__(self, other: object) -> bool:
            return False

    step, trace_case, trace, effect = _single_step_trace_fixture()

    def provider(_: TraceCase) -> TraceEvidence:
        misaligned_effect = PhysicalCallEffect(
            AlwaysEqualIndex(1),
            effect.base_call_fingerprint,
            effect.effect_digest,
        )
        return TraceEvidence(
            trace,
            trace,
            [step],
            1,
            1,
            [misaligned_effect],
            [misaligned_effect],
        )

    result = check_trace_contract(
        _IdentityAdapter(_variant()),
        [trace_case],
        provider,
        [step],
    )

    assert result.passed is False


def test_trace_provider_cannot_change_registered_step_content_in_place() -> None:
    variant = _variant()
    adapter = _IdentityAdapter(variant)
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    step = DenotationCase(
        "denotation-provider-mutation",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    replacement_action = SemanticAction("delete_all", {})
    replacement_call = {"name": "delete_all", "arguments": {}}
    replacement = DenotationCase(
        step.case_id,
        replacement_call,
        [replacement_action],
        [[replacement_call]],
        [[{"deleted": True}]],
        {"deleted": True},
    )

    def trace_provider(_: TraceCase) -> TraceEvidence:
        _replace_denotation_case_content(step, replacement)
        trace = ExecutionTrace(
            [step.surface_call],
            step.expected_actions,
            step.expected_base_call_groups[0],
        )
        effect = _effect(step.expected_base_call_groups[0][0])
        return TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [SchemaProbe("schema-provider-mutation", "search", call, [action])],
        [step],
        [StateCase("state-provider-mutation", "episode-provider-mutation")],
        lambda _: _state_evidence(),
        [TraceCase("trace-provider-mutation", "episode-provider-mutation")],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_PROVIDER_MUTATED_STEP")
    assert "trace.verified_step_mutated" in {
        item.code for item in suite.results[3].diagnostics
    }


def test_trace_provider_exception_still_checks_registered_step_content() -> None:
    step, trace_case, _, _ = _single_step_trace_fixture()
    replacement_call = {"name": "delete_all", "arguments": {}}
    replacement = DenotationCase(
        step.case_id,
        replacement_call,
        [SemanticAction("delete_all", {})],
        [[replacement_call]],
        [[{"deleted": True}]],
        {"deleted": True},
    )

    def provider(_: TraceCase) -> TraceEvidence:
        _replace_denotation_case_content(step, replacement)
        raise RuntimeError("provider payload /private/trace 779")

    result = check_trace_contract(
        _IdentityAdapter(_variant()),
        [trace_case],
        provider,
        [step],
    )
    codes = {item.code for item in result.diagnostics}
    diagnostic_text = " ".join(
        f"{item.code} {item.message} {item.case_id}" for item in result.diagnostics
    )

    assert codes >= {"trace.provider_exception", "trace.verified_step_mutated"}
    assert "/private" not in diagnostic_text
    assert "779" not in diagnostic_text


@pytest.mark.parametrize("mutated_layer", ["state", "trace"])
def test_suite_rejects_provider_mutation_of_its_case_content(
    mutated_layer: str,
) -> None:
    variant, adapter, probe, step, _, _, _ = _identity_suite_components()
    checked_state = StateCase("state-provider-case", "episode-a")
    replacement_state = StateCase("state-provider-case", "episode-b")
    checked_trace = TraceCase("trace-provider-case", "episode-a")
    replacement_trace = TraceCase("trace-provider-case", "episode-b")
    state_cases = [
        checked_state if mutated_layer == "state" else replacement_state
    ]
    trace_cases = [
        replacement_trace if mutated_layer == "state" else checked_trace
    ]

    def state_provider(_: StateCase) -> StateEvidence:
        if mutated_layer == "state":
            _replace_state_case_content(checked_state, replacement_state)
        return _state_evidence()

    def trace_provider(_: TraceCase) -> TraceEvidence:
        if mutated_layer == "trace":
            _replace_trace_case_content(checked_trace, replacement_trace)
        call = step.surface_call
        action = step.expected_actions[0]
        trace = ExecutionTrace([call], [action], [call])
        effect = _effect(call)
        return TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        state_cases,
        state_provider,
        trace_cases,
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail(f"ADMITTED_MUTATED_{mutated_layer.upper()}_CASE")
    layer_index = 2 if mutated_layer == "state" else 3
    assert "suite.case_snapshot_mismatch" in {
        item.code for item in suite.results[layer_index].diagnostics
    }


def test_suite_checks_initial_variant_binding_after_each_layer() -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    original_snapshot = SchemaVariant(
        variant.variant_id,
        variant.tools,
        variant.manifest,
    )
    intermediate = SchemaVariant(
        "intermediate-v1",
        variant.tools,
        {"operator": "intermediate", "seed": 11},
    )
    adapter = _LayerBoundaryVariantAdapter(variant, intermediate)

    def state_provider(_: StateCase) -> StateEvidence:
        _replace_variant_content(variant, intermediate)
        adapter.begin_intermediate_window()
        return _state_evidence()

    def trace_provider(_: TraceCase) -> TraceEvidence:
        _replace_variant_content(variant, original_snapshot)
        return trace_evidence

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        state_provider,
        [trace_case],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_BETWEEN_LAYER_VARIANT_CHANGE")
    assert "suite.variant_binding_mismatch" in {
        item.code for item in suite.results[2].diagnostics
    }


def test_binding_check_side_effect_cannot_follow_final_case_validation() -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    replacement_call = {"name": "delete_all", "arguments": {}}
    replacement = DenotationCase(
        step.case_id,
        replacement_call,
        [SemanticAction("delete_all", {})],
        [[replacement_call]],
        [[{"deleted": True}]],
        {"deleted": True},
    )

    class SideEffectingBindingAdapter(_IdentityAdapter):
        def __init__(self) -> None:
            super().__init__(variant)
            self.variant_reads_after_provider: int | None = None

        @property
        def variant(self) -> SchemaVariant:
            if self.variant_reads_after_provider is not None:
                self.variant_reads_after_provider += 1
                if self.variant_reads_after_provider == 3:
                    _replace_denotation_case_content(step, replacement)
                    self.variant_reads_after_provider = None
            return self._variant

    adapter = SideEffectingBindingAdapter()

    def trace_provider(_: TraceCase) -> TraceEvidence:
        adapter.variant_reads_after_provider = 0
        return trace_evidence

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_MUTATION_AFTER_FINAL_CASE_VALIDATION")
    assert "suite.case_snapshot_mismatch" in {
        item.code for item in suite.results[3].diagnostics
    }


def test_final_binding_check_uses_raw_adapter_slot_after_property_side_effect() -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    replacement = SchemaVariant(
        "getter-rebound-v1",
        variant.tools,
        {"operator": "getter-rebound", "seed": 14},
    )

    class SideEffectingGetterAdapter(_IdentityAdapter):
        def __init__(self) -> None:
            super().__init__(variant)
            self.variant_reads_after_provider: int | None = None

        @property
        def variant(self) -> SchemaVariant:
            current = self._variant
            if self.variant_reads_after_provider is not None:
                self.variant_reads_after_provider += 1
                if self.variant_reads_after_provider == 3:
                    self._variant = replacement
                    self.variant_reads_after_provider = None
            return current

    adapter = SideEffectingGetterAdapter()

    def trace_provider(_: TraceCase) -> TraceEvidence:
        adapter.variant_reads_after_provider = 0
        return trace_evidence

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_PROPERTY_REBOUND_ADAPTER")
    assert adapter._variant is replacement
    assert suite.passed is False


def test_suite_never_admits_an_entry_case_repaired_after_snapshot() -> None:
    variant = _variant()
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    step = DenotationCase(
        "denotation-invalid-at-entry",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    replacement = DenotationCase(
        step.case_id,
        step.surface_call,
        step.expected_actions,
        step.expected_base_call_groups,
        step.base_observation_groups,
        step.expected_surface_observation,
    )
    object.__setattr__(step, "_snapshot_fingerprint", "a" * 64)

    class RepairingAdapter(_IdentityAdapter):
        def __init__(self) -> None:
            super().__init__(variant)
            self._repaired = False

        def surface_to_semantic(
            self, surface_call: Mapping[str, JSONValue]
        ) -> tuple[SemanticAction, ...]:
            if not self._repaired:
                _replace_denotation_case_content(step, replacement)
                self._repaired = True
            return super().surface_to_semantic(surface_call)

    def trace_provider(_: TraceCase) -> TraceEvidence:
        trace = ExecutionTrace([call], [action], [call])
        effect = _effect(call)
        return TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])

    suite = evaluate_contract_suite(
        variant,
        RepairingAdapter(),
        [SchemaProbe("schema-repaired-entry", "search", call, [action])],
        [step],
        [StateCase("state-repaired-entry", "episode-repaired-entry")],
        lambda _: _state_evidence(),
        [TraceCase("trace-repaired-entry", "episode-repaired-entry")],
        trace_provider,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_REPAIRED_INVALID_ENTRY_CASE")
    assert "suite.case_snapshot_mismatch" in {
        item.code for item in suite.results[0].diagnostics
    }


def test_suite_never_admits_a_state_case_repaired_after_snapshot() -> None:
    variant, _, probe, step, _, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    state_case = StateCase("state-invalid-at-entry", trace_case.episode_id)
    replacement = StateCase(state_case.case_id, state_case.episode_id)
    object.__setattr__(state_case, "_snapshot_fingerprint", "a" * 64)

    class RepairingAdapter(_IdentityAdapter):
        def __init__(self) -> None:
            super().__init__(variant)
            self._repaired = False

        def surface_to_semantic(
            self, surface_call: Mapping[str, JSONValue]
        ) -> tuple[SemanticAction, ...]:
            if not self._repaired:
                _replace_state_case_content(state_case, replacement)
                self._repaired = True
            return super().surface_to_semantic(surface_call)

    suite = evaluate_contract_suite(
        variant,
        RepairingAdapter(),
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        lambda _: trace_evidence,
    )

    try:
        require_dataset_admission(variant, suite)
    except ValueError:
        pass
    else:
        pytest.fail("ADMITTED_REPAIRED_INVALID_STATE_CASE")
    assert "suite.case_snapshot_mismatch" in {
        item.code for item in suite.results[0].diagnostics
    }


def test_entry_snapshot_validation_checks_later_groups_after_invalid_entry() -> None:
    class Entry:
        _snapshot_fingerprint = "a" * 64

    entry = Entry()
    calls: list[object] = []

    def reject(_: object) -> object:
        raise ValueError("invalid entry")

    def record(value: object) -> object:
        calls.append(value)
        return value

    invalid_group = schema_contracts._validate_entry_case_group(
        schema_contracts._capture_entry_case_group(
            (object(),), Entry, lambda _: True, reject
        )
    )
    valid_group = schema_contracts._validate_entry_case_group(
        schema_contracts._capture_entry_case_group(
            (entry,), Entry, lambda _: True, record
        )
    )
    calls.clear()

    assert schema_contracts._entry_case_groups_match(
        (invalid_group, valid_group)
    ) is False
    assert calls == [entry]


def test_suite_runs_all_four_layers_when_every_checker_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant, adapter, probe, step, state_case, trace_case, _ = _identity_suite_components()
    calls: list[str] = []

    def exploding_checker(name: str):
        def raise_payload(*args: object) -> LayerContractResult:
            calls.append(name)
            raise RuntimeError(f"{name} payload /private/path 818")

        return raise_payload

    monkeypatch.setattr(
        schema_contracts, "check_schema_contract", exploding_checker("schema")
    )
    monkeypatch.setattr(
        denotation_contracts,
        "check_denotation_contract",
        exploding_checker("denotation"),
    )
    monkeypatch.setattr(state_contracts, "check_state_contract", exploding_checker("state"))
    monkeypatch.setattr(trace_contracts, "check_trace_contract", exploding_checker("trace"))

    suite = evaluate_contract_suite(
        variant,
        adapter,
        [probe],
        [step],
        [state_case],
        lambda _: _state_evidence(),
        [trace_case],
        lambda _: None,
    )
    diagnostic_text = " ".join(
        f"{item.code} {item.message} {item.case_id}"
        for result in suite.results
        for item in result.diagnostics
    )

    assert calls == ["schema", "denotation", "state", "trace"]
    assert suite.passed is False
    assert {result.layer for result in suite.results} == {
        "schema",
        "denotation",
        "state",
        "trace",
    }
    assert "/private" not in diagnostic_text
    assert "818" not in diagnostic_text


def test_suite_enforces_identical_state_and_trace_episode_sets() -> None:
    variant, suite = _evaluate_identity_components(
        state_episode="episode-state",
        trace_episode="episode-trace",
    )

    trace_result = suite.results[-1]
    assert "suite.episode_set_mismatch" in {
        item.code for item in trace_result.diagnostics
    }
    with pytest.raises(ValueError, match=r"pass|episode"):
        require_dataset_admission(variant, suite)


def test_hard_admission_rejects_bad_mapping_and_bad_evidence_providers() -> None:
    variant, _, _, step, _, _, trace_evidence = _identity_suite_components()
    _, bad_mapping_suite = _evaluate_identity_components(
        adapter_override=_BadSchemaAdapter(variant)
    )
    with pytest.raises(ValueError, match="pass"):
        require_dataset_admission(variant, bad_mapping_suite)

    def bad_state_provider(_: StateCase) -> StateEvidence:
        return _state_evidence(candidate_final_sha256="c" * 64)

    bad_effect = PhysicalCallEffect(
        0,
        trace_evidence.candidate_effects[0].base_call_fingerprint,
        "effect-extra",
    )
    bad_trace_evidence = TraceEvidence(
        trace_evidence.reference_trace,
        trace_evidence.candidate_trace,
        [step],
        {"score": True},
        {"score": 1},
        trace_evidence.reference_effects,
        [bad_effect],
    )
    variant, bad_evidence_suite = _evaluate_identity_components(
        state_provider_override=bad_state_provider,
        trace_evidence_override=bad_trace_evidence,
    )

    assert {item.code for item in bad_evidence_suite.results[2].diagnostics} >= {
        "state.final_mismatch"
    }
    assert {item.code for item in bad_evidence_suite.results[3].diagnostics} >= {
        "trace.effect_mismatch",
        "trace.score_mismatch",
    }
    with pytest.raises(ValueError, match="pass"):
        require_dataset_admission(variant, bad_evidence_suite)


def test_hard_admission_rejects_empty_vacuous_evidence() -> None:
    variant = _variant()
    suite = evaluate_contract_suite(
        variant,
        _IdentityAdapter(variant),
        [],
        [],
        [],
        lambda _: _state_evidence(),
        [],
        lambda _: None,
    )

    assert [result.checks_run for result in suite.results] == [0, 0, 0, 0]
    with pytest.raises(ValueError, match=r"non-vacuous|pass"):
        require_dataset_admission(variant, suite)


def test_suite_result_rejects_missing_and_duplicate_layers() -> None:
    fingerprint = "a" * 64
    results = [
        LayerContractResult(layer, fingerprint, 1, [])
        for layer in ("schema", "denotation", "state", "trace")
    ]

    with pytest.raises(ValueError, match="exactly one"):
        ContractSuiteResult(fingerprint, results[:-1])
    with pytest.raises(ValueError, match="exactly one"):
        ContractSuiteResult(fingerprint, [*results[:-1], results[0]])


def test_hard_admission_rejects_forged_and_object_setattr_mutated_results() -> None:
    variant = _variant()
    schema_digest = schema_fingerprint(variant)
    forged_results = [
        LayerContractResult(
            layer,
            schema_digest if layer == "schema" else "a" * 64,
            1,
            [],
        )
        for layer in ("schema", "denotation", "state", "trace")
    ]
    forged_suite = ContractSuiteResult(schema_digest, forged_results)
    with pytest.raises(ValueError, match="attested"):
        require_dataset_admission(variant, forged_suite)

    variant, mutated_suite = _evaluate_identity_components()
    object.__setattr__(mutated_suite.results[1], "checks_run", True)
    with pytest.raises(ValueError, match=r"checks_run|mutated"):
        require_dataset_admission(variant, mutated_suite)

    variant, mutated_fingerprint_suite = _evaluate_identity_components()
    object.__setattr__(mutated_fingerprint_suite, "fingerprint", "f" * 64)
    with pytest.raises(ValueError, match=r"fingerprint|attested"):
        require_dataset_admission(variant, mutated_fingerprint_suite)

    variant, duplicate_suite = _evaluate_identity_components()
    object.__setattr__(
        duplicate_suite,
        "results",
        (
            duplicate_suite.results[0],
            duplicate_suite.results[0],
            duplicate_suite.results[2],
            duplicate_suite.results[3],
        ),
    )
    with pytest.raises(ValueError, match=r"exactly one|fingerprint|order|mutated"):
        require_dataset_admission(variant, duplicate_suite)

    variant, reordered_suite = _evaluate_identity_components()
    object.__setattr__(
        reordered_suite,
        "results",
        (
            reordered_suite.results[0],
            reordered_suite.results[2],
            reordered_suite.results[1],
            reordered_suite.results[3],
        ),
    )
    with pytest.raises(ValueError, match=r"order|mutated"):
        require_dataset_admission(variant, reordered_suite)


def test_hard_admission_rejects_copied_private_seals_on_forged_suite() -> None:
    variant, genuine = _evaluate_identity_components()
    forged_results: list[LayerContractResult] = []
    for original in genuine.results:
        forged = LayerContractResult(
            original.layer,
            original.fingerprint,
            original.checks_run,
            original.diagnostics,
        )
        object.__setattr__(forged, "_attestation", original._attestation)
        object.__setattr__(
            forged,
            "_sealed_snapshot_fingerprint",
            forged._snapshot_fingerprint,
        )
        forged_results.append(forged)
    forged_suite = ContractSuiteResult(genuine.schema_fingerprint, forged_results)
    object.__setattr__(forged_suite, "_attestation", genuine._attestation)
    object.__setattr__(forged_suite, "_sealed_fingerprint", forged_suite.fingerprint)
    object.__setattr__(
        forged_suite,
        "_state_episode_fingerprint",
        genuine._state_episode_fingerprint,
    )
    object.__setattr__(
        forged_suite,
        "_trace_episode_fingerprint",
        genuine._trace_episode_fingerprint,
    )

    with pytest.raises(ValueError, match=r"evaluation|issued|provenance|forged"):
        require_dataset_admission(variant, forged_suite)


def test_hard_admission_uses_external_provenance_for_evaluator_failure() -> None:
    supplied_variant = _variant()
    variant, suite = _evaluate_identity_components(
        adapter_override=_BadSchemaAdapter(supplied_variant)
    )
    assert suite.passed is False

    for result in suite.results:
        clean = LayerContractResult(
            result.layer,
            result.fingerprint,
            result.checks_run,
            [],
        )
        object.__setattr__(result, "diagnostics", ())
        object.__setattr__(
            result,
            "_snapshot_fingerprint",
            clean._snapshot_fingerprint,
        )
        object.__setattr__(
            result,
            "_sealed_snapshot_fingerprint",
            clean._snapshot_fingerprint,
        )
    aggregate = contract_suite_fingerprint(suite.schema_fingerprint, suite.results)
    object.__setattr__(suite, "fingerprint", aggregate)
    object.__setattr__(suite, "_sealed_fingerprint", aggregate)
    object.__setattr__(
        suite,
        "_admission_snapshot_fingerprint",
        schema_contracts._admission_snapshot_fingerprint(suite),
    )
    with pytest.raises(ValueError, match=r"provenance|mutated"):
        _ = suite.passed

    with pytest.raises(ValueError, match=r"provenance|mutated|issued"):
        require_dataset_admission(variant, suite)


def test_external_provenance_rejects_nonexact_digest_values() -> None:
    supplied_variant = _variant()
    variant, suite = _evaluate_identity_components(
        adapter_override=_BadSchemaAdapter(supplied_variant)
    )
    assert suite.passed is False

    clean_results = tuple(
        LayerContractResult(
            result.layer,
            result.fingerprint,
            result.checks_run,
            [],
        )
        for result in suite.results
    )
    aggregate = contract_suite_fingerprint(suite.schema_fingerprint, clean_results)
    for result, clean in zip(suite.results, clean_results, strict=True):
        disguised_snapshot = _AlwaysEqualDigest(clean._snapshot_fingerprint)
        object.__setattr__(result, "diagnostics", ())
        object.__setattr__(result, "_snapshot_fingerprint", disguised_snapshot)
        object.__setattr__(
            result,
            "_sealed_snapshot_fingerprint",
            disguised_snapshot,
        )
    object.__setattr__(suite, "fingerprint", aggregate)
    admission_snapshot = schema_contracts._admission_snapshot_fingerprint(suite)
    disguised_aggregate = _AlwaysEqualDigest(aggregate)
    object.__setattr__(suite, "fingerprint", disguised_aggregate)
    object.__setattr__(suite, "_sealed_fingerprint", disguised_aggregate)
    object.__setattr__(
        suite,
        "_admission_snapshot_fingerprint",
        _AlwaysEqualDigest(admission_snapshot),
    )

    with pytest.raises(ValueError, match=r"digest|fingerprint|provenance|mutated"):
        _ = suite.passed
    with pytest.raises(ValueError, match=r"digest|fingerprint|provenance|mutated"):
        require_dataset_admission(variant, suite)


def test_public_revalidation_rejects_tuple_subclasses_before_iteration() -> None:
    class ExplodingTuple(tuple):
        def __iter__(self):
            raise RuntimeError("tuple payload /private/container 778")

    result = LayerContractResult("schema", "a" * 64, 1, [])
    object.__setattr__(result, "diagnostics", ExplodingTuple())
    with pytest.raises(ValueError) as layer_error:
        _ = result.passed
    assert "/private" not in str(layer_error.value)
    assert "778" not in str(layer_error.value)

    _, suite = _evaluate_identity_components()
    object.__setattr__(suite, "results", ExplodingTuple(suite.results))
    with pytest.raises(ValueError) as suite_error:
        _ = suite.passed
    assert "/private" not in str(suite_error.value)
    assert "778" not in str(suite_error.value)


def test_suite_passed_revalidates_object_setattr_mutation() -> None:
    supplied_variant = _variant()
    _, suite = _evaluate_identity_components(
        adapter_override=_BadSchemaAdapter(supplied_variant)
    )
    assert suite.passed is False
    for result in suite.results:
        object.__setattr__(result, "diagnostics", ())

    with pytest.raises(ValueError, match=r"provenance|mutated"):
        _ = suite.passed


def test_hard_admission_rejects_dual_episode_fingerprint_mutation() -> None:
    variant, suite = _evaluate_identity_components()
    object.__setattr__(suite, "_state_episode_fingerprint", "e" * 64)
    object.__setattr__(suite, "_trace_episode_fingerprint", "e" * 64)

    with pytest.raises(ValueError, match=r"episode|provenance|mutated"):
        require_dataset_admission(variant, suite)


def test_hard_admission_recomputes_variant_schema_fingerprint() -> None:
    _, suite = _evaluate_identity_components()
    other_variant = SchemaVariant(
        "other-v1",
        [_tool()],
        {"operator": "identity", "seed": 7},
    )

    with pytest.raises(ValueError, match="schema fingerprint"):
        require_dataset_admission(other_variant, suite)
