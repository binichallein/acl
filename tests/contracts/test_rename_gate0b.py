"""Synthetic four-layer Gate 0b coverage for pure tool-name rename."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pytest

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts import (
    DenotationCase,
    PhysicalCallEffect,
    SchemaProbe,
    StateCase,
    StateEvidence,
    TraceCase,
    TraceEvidence,
    base_call_fingerprint,
    evaluate_contract_suite,
    require_dataset_admission,
)
from toolshift.transforms.rename import apply_rename
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
)


def _tool(name: str) -> SurfaceToolSpec:
    return SurfaceToolSpec(
        name,
        f"Synthetic {name} tool",
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )


def _base_variant() -> SchemaVariant:
    return SchemaVariant(
        "synthetic-base-v1",
        (_tool("search"), _tool("summarize")),
        {"operator": "identity", "seed": 11},
    )


class _SyntheticIdentityAdapter(SemanticAdapter):
    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        name = surface_call.get("name")
        arguments = surface_call.get("arguments")
        if type(name) is not str or name not in {tool.name for tool in self.variant.tools}:
            raise ValueError("unknown synthetic tool")
        if not isinstance(arguments, Mapping):
            raise ValueError("synthetic arguments must be a mapping")
        return (SemanticAction(name, arguments),)

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        return ({"name": action.name, "arguments": action.arguments},)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        return trace.semantic_actions


class _WrongDenotationSource(_SyntheticIdentityAdapter):
    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        actions = super().surface_to_semantic(surface_call)
        if actions[0].name == "search":
            return (SemanticAction("summarize", actions[0].arguments),)
        return actions


def _state_evidence() -> StateEvidence:
    initial = hashlib.sha256(b"synthetic-initial").hexdigest()
    final = hashlib.sha256(b"synthetic-final").hexdigest()
    return StateEvidence(
        initial,
        initial,
        final,
        final,
        initial,
        initial,
        "synthetic-collateral",
        "synthetic-collateral",
    )


def _effect(call: Mapping[str, JSONValue], index: int) -> PhysicalCallEffect:
    return PhysicalCallEffect(
        index,
        base_call_fingerprint(call),
        f"synthetic-effect-{index}",
    )


def _suite_inputs(source_type: type[SemanticAdapter] = _SyntheticIdentityAdapter):
    base = _base_variant()
    adapter = apply_rename(
        source_type(base),
        tool_name_mapping={"search": "lookup"},
        seed=17,
    )
    lookup_call = {"name": "lookup", "arguments": {"value": "alpha"}}
    canonical_call = {"name": "search", "arguments": {"value": "alpha"}}
    summarize_call = {"name": "summarize", "arguments": {"value": "alpha"}}
    search_action = SemanticAction("search", {"value": "alpha"})
    summarize_action = SemanticAction("summarize", {"value": "alpha"})
    lookup_observation = {"items": ["alpha"]}
    summarize_observation = {"summary": "alpha"}
    probes = (
        SchemaProbe("schema-lookup", "lookup", lookup_call, (search_action,)),
        SchemaProbe(
            "schema-summarize",
            "summarize",
            summarize_call,
            (summarize_action,),
        ),
    )
    lookup_denotation = DenotationCase(
        "denotation-lookup",
        lookup_call,
        (search_action,),
        ((canonical_call,),),
        ((lookup_observation,),),
        lookup_observation,
    )
    summarize_denotation = DenotationCase(
        "denotation-summarize",
        summarize_call,
        (summarize_action,),
        ((summarize_call,),),
        ((summarize_observation,),),
        summarize_observation,
    )
    reference_trace = ExecutionTrace(
        (canonical_call, summarize_call),
        (search_action, summarize_action),
        (canonical_call, summarize_call),
    )
    candidate_trace = ExecutionTrace(
        (lookup_call, summarize_call),
        (search_action, summarize_action),
        (canonical_call, summarize_call),
    )
    trace_evidence = TraceEvidence(
        reference_trace,
        candidate_trace,
        (lookup_denotation, summarize_denotation),
        {"success": True, "score": 1},
        {"success": True, "score": 1},
        tuple(_effect(call, index) for index, call in enumerate(reference_trace.base_calls)),
        tuple(_effect(call, index) for index, call in enumerate(candidate_trace.base_calls)),
    )
    return adapter, probes, (lookup_denotation, summarize_denotation), trace_evidence


def test_synthetic_rename_passes_all_four_layers_and_hard_gate_immediately() -> None:
    adapter, probes, denotations, trace_evidence = _suite_inputs()
    state_case = StateCase("state-rename", "synthetic-episode")
    trace_case = TraceCase("trace-rename", "synthetic-episode")
    assert tuple(tool.name for tool in adapter.variant.tools) == ("lookup", "summarize")
    assert {probe.surface_tool_name for probe in probes} == {
        tool.name for tool in adapter.variant.tools
    }
    assert len(denotations) == len(trace_evidence.candidate_steps) == 2
    assert all(
        candidate_step is denotation
        for candidate_step, denotation in zip(
            trace_evidence.candidate_steps,
            denotations,
            strict=True,
        )
    )
    assert trace_evidence.reference_trace.base_calls == trace_evidence.candidate_trace.base_calls
    assert trace_evidence.reference_score == trace_evidence.candidate_score
    assert trace_evidence.reference_effects[0].effect_digest == (
        trace_evidence.candidate_effects[0].effect_digest
    )

    suite = evaluate_contract_suite(
        variant=adapter.variant,
        adapter=adapter,
        schema_probes=probes,
        denotation_cases=denotations,
        state_cases=(state_case,),
        state_provider=lambda requested: _state_evidence(),
        trace_cases=(trace_case,),
        trace_provider=lambda requested: trace_evidence,
    )
    admitted = require_dataset_admission(adapter.variant, suite)

    assert admitted is suite
    assert suite.passed is True
    assert [result.layer for result in suite.results] == [
        "schema",
        "denotation",
        "state",
        "trace",
    ]
    assert all(result.checks_run > 0 for result in suite.results)


def test_synthetic_gate_rejects_incorrect_denotation_after_valid_construction() -> None:
    adapter, probes, denotations, trace_evidence = _suite_inputs(_WrongDenotationSource)
    suite = evaluate_contract_suite(
        variant=adapter.variant,
        adapter=adapter,
        schema_probes=probes,
        denotation_cases=denotations,
        state_cases=(StateCase("state-bad", "synthetic-bad-episode"),),
        state_provider=lambda requested: _state_evidence(),
        trace_cases=(TraceCase("trace-bad", "synthetic-bad-episode"),),
        trace_provider=lambda requested: trace_evidence,
    )

    assert [result.layer for result in suite.results] == [
        "schema",
        "denotation",
        "state",
        "trace",
    ]
    assert suite.passed is False
    diagnostic_codes = {
        diagnostic.code for result in suite.results[:2] for diagnostic in result.diagnostics
    }
    assert "schema.mapping_mismatch" in diagnostic_codes
    assert "denotation.parse_mismatch" in diagnostic_codes
    with pytest.raises(ValueError, match="pass"):
        require_dataset_admission(adapter.variant, suite)
