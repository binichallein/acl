"""Trace behavioral-equivalence contract tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pytest

import toolshift.contracts._common as common_contracts
from toolshift.contracts.denotation import DenotationCase
from toolshift.contracts.schema import (
    SchemaProbe,
    evaluate_contract_suite,
    require_dataset_admission,
)
from toolshift.contracts.state import StateCase
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
    canonical_json_bytes,
)

from ._support import (
    _effect,
    _IdentityAdapter,
    _NoOpAdapter,
    _single_step_trace_fixture,
    _state_evidence,
    _surface_call,
    _variant,
)


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
    assert "trace.empty_steps" in {item.code for item in empty_suite.results[3].diagnostics}
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

    result = check_trace_contract(TraceOnlyAdapter(_variant()), [case], lambda _: evidence, [step])

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

    assert "trace.canonical_semantic_channel_mismatch" in {item.code for item in result.diagnostics}


def test_trace_rejects_effect_score_and_alignment_mismatches_type_sensitively() -> None:
    step, case, trace, effect = _single_step_trace_fixture()
    bad_candidate_effect = PhysicalCallEffect(0, effect.base_call_fingerprint, "effect-extra")
    evidence = TraceEvidence(
        trace,
        trace,
        [step],
        {"score": True},
        {"score": 1},
        [effect],
        [bad_candidate_effect],
    )

    result = check_trace_contract(_IdentityAdapter(_variant()), [case], lambda _: evidence, [step])

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

    result = check_trace_contract(_IdentityAdapter(_variant()), [case], lambda _: evidence, [step])

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


def test_trace_provider_failure_uses_the_stable_error_fingerprint_marker() -> None:
    step, case, _, _ = _single_step_trace_fixture()

    def provider(_: TraceCase) -> TraceEvidence:
        raise RuntimeError("provider failure")

    result = check_trace_contract(
        _IdentityAdapter(_variant()),
        [case],
        provider,
        [step],
    )

    assert [item.code for item in result.diagnostics] == ["trace.provider_exception"]
    assert result.fingerprint == common_contracts._fingerprint_parts(
        b"toolshift.trace-suite.v1",
        [bytes.fromhex(case._snapshot_fingerprint), b"provider-error"],
    )


def test_trace_reference_canonicalization_failure_still_checks_binding_once() -> None:
    class ReferenceFailureAdapter(_IdentityAdapter):
        def __init__(self) -> None:
            super().__init__(_variant())
            self.variant_reads = 0
            self.canonical_calls: list[ExecutionTrace] = []

        @property
        def variant(self) -> SchemaVariant:
            self.variant_reads += 1
            return self._variant

        def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
            self.canonical_calls.append(trace)
            raise RuntimeError("canonicalization failure")

    step, case, trace, effect = _single_step_trace_fixture()
    evidence = TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])
    adapter = ReferenceFailureAdapter()

    result = check_trace_contract(adapter, [case], lambda _: evidence, [step])

    assert [item.code for item in result.diagnostics] == ["trace.canonicalization_exception"]
    assert len(adapter.canonical_calls) == 1
    assert adapter.canonical_calls[0] is evidence.reference_trace
    assert adapter.variant_reads == 2


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

    assert "trace.provider_exception" in {item.code for item in evidence_result.diagnostics}
    assert "trace.invalid_case" in {item.code for item in case_result.diagnostics}
