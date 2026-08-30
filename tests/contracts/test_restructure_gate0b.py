"""Synthetic four-layer Gate 0b coverage for L2 parameter restructuring."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pytest

import toolshift.transforms.restructure as restructure_module
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
from toolshift.transforms.restructure import ParameterGroupRule
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
)


def _base_variant() -> SchemaVariant:
    return SchemaVariant(
        "synthetic-restructure-base-v1",
        (
            SurfaceToolSpec(
                "search",
                "Synthetic search tool",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "locale": {"type": "string"},
                    },
                    "required": ("query",),
                    "additionalProperties": False,
                },
            ),
            SurfaceToolSpec(
                "summarize",
                "Synthetic summarize tool",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ("value",),
                    "additionalProperties": False,
                },
            ),
        ),
        {"operator": "identity", "seed": 11},
    )


class _SyntheticRestructureSource(SemanticAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._tool_names = frozenset(tool.name for tool in variant.tools)

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        name = surface_call.get("name")
        arguments = surface_call.get("arguments")
        if type(name) is not str or name not in self._tool_names:
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


class _WrongSearchMappingSource(_SyntheticRestructureSource):
    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        actions = super().surface_to_semantic(surface_call)
        if actions[0].name == "search":
            return (SemanticAction("summarize", actions[0].arguments),)
        return actions


def _state_evidence() -> StateEvidence:
    initial = hashlib.sha256(b"synthetic-restructure-initial").hexdigest()
    final = hashlib.sha256(b"synthetic-restructure-final").hexdigest()
    return StateEvidence(
        initial,
        initial,
        final,
        final,
        initial,
        initial,
        "synthetic-restructure-collateral",
        "synthetic-restructure-collateral",
    )


def _effect(call: Mapping[str, JSONValue], index: int) -> PhysicalCallEffect:
    return PhysicalCallEffect(
        index,
        base_call_fingerprint(call),
        f"synthetic-restructure-effect-{index}",
    )


def _suite_inputs(source_type: type[SemanticAdapter] = _SyntheticRestructureSource):
    base = _base_variant()
    adapter = restructure_module.apply_parameter_restructure(
        source_type(base),
        rules=(ParameterGroupRule("search", "request", ("query", "limit")),),
        seed=17,
    )
    surface_search = {
        "name": "search",
        "arguments": {
            "request": {"query": "alpha", "limit": 2},
            "locale": "en",
        },
    }
    canonical_search = {
        "name": "search",
        "arguments": {"query": "alpha", "limit": 2, "locale": "en"},
    }
    summarize_call = {"name": "summarize", "arguments": {"value": "alpha"}}
    search_action = SemanticAction("search", canonical_search["arguments"])
    summarize_action = SemanticAction("summarize", summarize_call["arguments"])
    search_observation = {"items": ["alpha"], "count": 1}
    summarize_observation = {"summary": "alpha"}
    probes = (
        SchemaProbe("schema-search", "search", surface_search, (search_action,)),
        SchemaProbe(
            "schema-summarize",
            "summarize",
            summarize_call,
            (summarize_action,),
        ),
    )
    search_denotation = DenotationCase(
        "denotation-search",
        surface_search,
        (search_action,),
        ((canonical_search,),),
        ((search_observation,),),
        search_observation,
    )
    summarize_denotation = DenotationCase(
        "denotation-summarize",
        summarize_call,
        (summarize_action,),
        ((summarize_call,),),
        ((summarize_observation,),),
        summarize_observation,
    )
    denotations = (search_denotation, summarize_denotation)
    reference_trace = ExecutionTrace(
        (canonical_search, summarize_call),
        (search_action, summarize_action),
        (canonical_search, summarize_call),
    )
    candidate_trace = ExecutionTrace(
        (surface_search, summarize_call),
        (search_action, summarize_action),
        (canonical_search, summarize_call),
    )
    trace_evidence = TraceEvidence(
        reference_trace,
        candidate_trace,
        denotations,
        {"success": True, "score": 1},
        {"success": True, "score": 1},
        tuple(_effect(call, index) for index, call in enumerate(reference_trace.base_calls)),
        tuple(_effect(call, index) for index, call in enumerate(candidate_trace.base_calls)),
    )
    state_case = StateCase("state-restructure", "synthetic-restructure-episode")
    trace_case = TraceCase("trace-restructure", "synthetic-restructure-episode")
    return (
        adapter,
        probes,
        denotations,
        state_case,
        _state_evidence(),
        trace_case,
        trace_evidence,
    )


def test_synthetic_restructure_passes_all_four_layers_and_hard_gate_immediately() -> None:
    (
        adapter,
        probes,
        denotations,
        state_case,
        state_evidence,
        trace_case,
        trace_evidence,
    ) = _suite_inputs()

    assert tuple(tool.name for tool in adapter.variant.tools) == ("search", "summarize")
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
    assert trace_evidence.reference_score == trace_evidence.candidate_score
    assert len(trace_evidence.reference_effects) == len(trace_evidence.candidate_effects) == 2
    for reference, candidate in zip(
        trace_evidence.reference_effects,
        trace_evidence.candidate_effects,
        strict=True,
    ):
        assert reference.call_index == candidate.call_index
        assert reference.base_call_fingerprint == candidate.base_call_fingerprint
        assert reference.effect_digest == candidate.effect_digest
    assert state_evidence.reference_initial_sha256 == state_evidence.candidate_initial_sha256
    assert state_evidence.reference_final_sha256 == state_evidence.candidate_final_sha256
    assert state_evidence.reference_post_reset_sha256 == (state_evidence.reference_initial_sha256)
    assert state_evidence.candidate_post_reset_sha256 == (state_evidence.candidate_initial_sha256)
    assert state_evidence.reference_collateral_digest == (
        state_evidence.candidate_collateral_digest
    )
    assert state_case.episode_id == trace_case.episode_id

    def state_provider(requested: StateCase) -> StateEvidence:
        assert requested is state_case
        return state_evidence

    def trace_provider(requested: TraceCase) -> TraceEvidence:
        assert requested is trace_case
        return trace_evidence

    suite = evaluate_contract_suite(
        variant=adapter.variant,
        adapter=adapter,
        schema_probes=probes,
        denotation_cases=denotations,
        state_cases=(state_case,),
        state_provider=state_provider,
        trace_cases=(trace_case,),
        trace_provider=trace_provider,
    )
    admitted = require_dataset_admission(adapter.variant, suite)

    assert admitted is suite
    assert suite.passed is True
    assert tuple(result.layer for result in suite.results) == (
        "schema",
        "denotation",
        "state",
        "trace",
    )
    assert tuple(result.checks_run for result in suite.results) == (2, 2, 1, 2)
    assert all(result.checks_run > 0 for result in suite.results)
    assert all(result.diagnostics == () for result in suite.results)


def test_synthetic_gate_rejects_wrong_search_mapping_after_valid_construction() -> None:
    (
        adapter,
        probes,
        denotations,
        state_case,
        state_evidence,
        trace_case,
        trace_evidence,
    ) = _suite_inputs(_WrongSearchMappingSource)

    suite = evaluate_contract_suite(
        variant=adapter.variant,
        adapter=adapter,
        schema_probes=probes,
        denotation_cases=denotations,
        state_cases=(state_case,),
        state_provider=lambda requested: state_evidence,
        trace_cases=(trace_case,),
        trace_provider=lambda requested: trace_evidence,
    )

    assert tuple(result.layer for result in suite.results) == (
        "schema",
        "denotation",
        "state",
        "trace",
    )
    assert suite.passed is False
    schema_codes = {diagnostic.code for diagnostic in suite.results[0].diagnostics}
    denotation_codes = {diagnostic.code for diagnostic in suite.results[1].diagnostics}
    assert "schema.mapping_mismatch" in schema_codes
    assert "denotation.parse_mismatch" in denotation_codes
    with pytest.raises(ValueError, match="pass"):
        require_dataset_admission(adapter.variant, suite)
