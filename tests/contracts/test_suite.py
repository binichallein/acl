"""Suite behavioral-equivalence contract tests."""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import UserList
from collections.abc import Callable, Mapping
from inspect import getdoc
from pathlib import Path
from types import MappingProxyType
from typing import get_type_hints

import pytest

import toolshift.contracts._schema as schema_core_contracts
import toolshift.contracts.denotation as denotation_contracts
import toolshift.contracts.schema as schema_contracts
import toolshift.contracts.state as state_contracts
import toolshift.contracts.suite as suite_contracts
import toolshift.contracts.trace as trace_contracts
import toolshift.types as types_module
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
)

from ._support import (
    _AlwaysEqualDigest,
    _BadSchemaAdapter,
    _DenotationListSwapAdapter,
    _effect,
    _evaluate_identity_components,
    _identity_suite_components,
    _IdentityAdapter,
    _InPlaceVariantMutationAdapter,
    _LayerBoundaryVariantAdapter,
    _replace_denotation_case_content,
    _replace_state_case_content,
    _replace_trace_case_content,
    _replace_variant_content,
    _single_step_trace_fixture,
    _StageRebindingAdapter,
    _state_evidence,
    _surface_call,
    _tool,
    _TransientStageRebindingAdapter,
    _variant,
)


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


@pytest.mark.parametrize("stage", ["parse", "compile", "wrap", "canonicalize"])
def test_suite_rejects_adapter_rebinding_during_every_adapter_stage(stage: str) -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = _identity_suite_components()
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

    assert "suite.variant_binding_mismatch" in {item.code for item in suite.results[0].diagnostics}
    assert suite.passed is False
    with pytest.raises(ValueError, match=r"pass|binding"):
        require_dataset_admission(variant, suite)


def test_suite_uses_initial_fingerprint_against_in_place_variant_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = _identity_suite_components()
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
    assert "suite.variant_binding_mismatch" in {item.code for item in suite.results[0].diagnostics}
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
    variant, _, probe, step, state_case, trace_case, trace_evidence = _identity_suite_components()
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
        diagnostic.code for result in suite.results for diagnostic in result.diagnostics
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
    assert "trace.unverified_step" in {item.code for item in suite.results[3].diagnostics}


def test_suite_rejects_sequence_subclass_before_it_can_change_entry_baseline() -> None:
    variant, adapter, probe, step, state_case, trace_case, _ = _identity_suite_components()
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
    variant, adapter, probe, step, state_case, trace_case, _ = _identity_suite_components()
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

    assert types_module._semantic_action_has_canonical_shape(action) is False
    assert callback_arguments.fired is False


def test_identity_suite_never_calls_rebound_frozen_mapping_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant, adapter, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components()
    )
    mapping_type = type(step.surface_call)
    callback_calls = 0

    def callback(_: object) -> tuple[tuple[str, JSONValue], ...]:
        nonlocal callback_calls
        callback_calls += 1
        raise AssertionError("suite validation called a rebound mapping method")

    monkeypatch.setattr(mapping_type, "_raw_items", callback)

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
    admitted = require_dataset_admission(variant, suite)

    assert callback_calls == 0
    assert suite.passed is True
    assert admitted is suite


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
    assert "suite.episode_set_mismatch" in {item.code for item in suite.results[3].diagnostics}


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

    variant, adapter, probe, step, _, _, trace_evidence = _identity_suite_components()
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
    assert "suite.case_snapshot_mismatch" in {item.code for item in suite.results[2].diagnostics}


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
    assert "suite.case_snapshot_mismatch" in {item.code for item in suite.results[2].diagnostics}


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
    assert "trace.verified_step_mutated" in {item.code for item in suite.results[3].diagnostics}


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
    state_cases = [checked_state if mutated_layer == "state" else replacement_state]
    trace_cases = [replacement_trace if mutated_layer == "state" else checked_trace]

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
    variant, _, probe, step, state_case, trace_case, trace_evidence = _identity_suite_components()
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
    assert "suite.variant_binding_mismatch" in {item.code for item in suite.results[2].diagnostics}


def test_binding_check_side_effect_cannot_follow_final_case_validation() -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = _identity_suite_components()
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
    assert "suite.case_snapshot_mismatch" in {item.code for item in suite.results[3].diagnostics}


def test_final_binding_check_uses_raw_adapter_slot_after_property_side_effect() -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = _identity_suite_components()
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
    assert "suite.case_snapshot_mismatch" in {item.code for item in suite.results[0].diagnostics}


def test_suite_never_admits_a_state_case_repaired_after_snapshot() -> None:
    variant, _, probe, step, _, trace_case, trace_evidence = _identity_suite_components()
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
    assert "suite.case_snapshot_mismatch" in {item.code for item in suite.results[0].diagnostics}


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

    invalid_group = suite_contracts._validate_entry_case_group(
        suite_contracts._capture_entry_case_group((object(),), Entry, lambda _: True, reject)
    )
    valid_group = suite_contracts._validate_entry_case_group(
        suite_contracts._capture_entry_case_group((entry,), Entry, lambda _: True, record)
    )
    calls.clear()

    assert suite_contracts._entry_case_groups_match((invalid_group, valid_group)) is False
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

    monkeypatch.setattr(schema_core_contracts, "check_schema_contract", exploding_checker("schema"))
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
    assert "suite.episode_set_mismatch" in {item.code for item in trace_result.diagnostics}
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
        suite_contracts._admission_snapshot_fingerprint(suite),
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
    admission_snapshot = suite_contracts._admission_snapshot_fingerprint(suite)
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


@pytest.mark.parametrize("container_kind", ["user-list", "list-subclass"])
def test_contract_suite_fingerprint_rejects_nonexact_result_containers_without_iteration(
    container_kind: str,
) -> None:
    result = LayerContractResult("schema", "a" * 64, 1, [])

    class ExplodingList(list[LayerContractResult]):
        def __iter__(self):
            raise AssertionError("nonexact result container was iterated")

    results: object = (
        UserList([result]) if container_kind == "user-list" else ExplodingList([result])
    )

    with pytest.raises(ValueError, match="exact built-in list or tuple"):
        contract_suite_fingerprint(
            "a" * 64,
            results,  # type: ignore[arg-type]
        )


def test_suite_passed_revalidates_object_setattr_mutation() -> None:
    supplied_variant = _variant()
    _, suite = _evaluate_identity_components(adapter_override=_BadSchemaAdapter(supplied_variant))
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


@pytest.mark.parametrize("container_kind", ["user-list", "list-subclass"])
@pytest.mark.parametrize(
    ("layer", "invalid_code"),
    [
        ("schema", "schema.invalid_probes"),
        ("denotation", "denotation.invalid_cases"),
        ("state", "state.invalid_cases"),
        ("trace", "trace.invalid_cases"),
    ],
)
def test_direct_checkers_reject_nonexact_outer_case_containers_without_iteration(
    container_kind: str,
    layer: str,
    invalid_code: str,
) -> None:
    variant, _, probe, step, state_case, trace_case, trace_evidence = _identity_suite_components()
    adapter_calls: list[str] = []

    class CountingAdapter(_IdentityAdapter):
        def surface_to_semantic(
            self,
            surface_call: Mapping[str, JSONValue],
        ) -> tuple[SemanticAction, ...]:
            adapter_calls.append("parse")
            return super().surface_to_semantic(surface_call)

        def semantic_to_base_calls(
            self,
            action: SemanticAction,
        ) -> tuple[Mapping[str, JSONValue], ...]:
            adapter_calls.append("compile")
            return super().semantic_to_base_calls(action)

        def base_observation_to_surface(
            self,
            surface_call: Mapping[str, JSONValue],
            actions: tuple[SemanticAction, ...],
            base_observation_groups: tuple[tuple[JSONValue, ...], ...],
        ) -> JSONValue:
            adapter_calls.append("wrap")
            return super().base_observation_to_surface(
                surface_call,
                actions,
                base_observation_groups,
            )

        def canonicalize_trace(
            self,
            trace: ExecutionTrace,
        ) -> tuple[SemanticAction, ...]:
            adapter_calls.append("canonicalize")
            return super().canonicalize_trace(trace)

    adapter = CountingAdapter(variant)

    class ExplodingList(list[object]):
        def __iter__(self):
            raise AssertionError("nonexact list container was iterated")

    def nonexact(value: object) -> object:
        if container_kind == "user-list":
            return UserList([value])
        return ExplodingList([value])

    provider_calls = 0

    def state_provider(_: StateCase) -> StateEvidence:
        nonlocal provider_calls
        provider_calls += 1
        return _state_evidence()

    def trace_provider(_: TraceCase) -> TraceEvidence:
        nonlocal provider_calls
        provider_calls += 1
        return trace_evidence

    checkers: dict[str, Callable[[], LayerContractResult]] = {
        "schema": lambda: check_schema_contract(
            variant,
            adapter,
            nonexact(probe),  # type: ignore[arg-type]
        ),
        "denotation": lambda: check_denotation_contract(
            adapter,
            nonexact(step),  # type: ignore[arg-type]
        ),
        "state": lambda: check_state_contract(
            nonexact(state_case),  # type: ignore[arg-type]
            state_provider,
        ),
        "trace": lambda: check_trace_contract(
            adapter,
            nonexact(trace_case),  # type: ignore[arg-type]
            trace_provider,
            [step],
        ),
    }

    result = checkers[layer]()

    assert invalid_code in {item.code for item in result.diagnostics}
    assert provider_calls == 0
    assert adapter_calls == []


@pytest.mark.parametrize("container_kind", ["user-list", "list-subclass"])
def test_trace_checker_rejects_nonexact_verified_step_container_without_iteration(
    container_kind: str,
) -> None:
    _, adapter, _, step, _, trace_case, trace_evidence = _identity_suite_components()

    class ExplodingList(list[DenotationCase]):
        def __iter__(self):
            raise AssertionError("nonexact verified-step container was iterated")

    verified_steps: object = (
        UserList([step]) if container_kind == "user-list" else ExplodingList([step])
    )
    provider_calls = 0

    def provider(_: TraceCase) -> TraceEvidence:
        nonlocal provider_calls
        provider_calls += 1
        return trace_evidence

    result = check_trace_contract(
        adapter,
        [trace_case],
        provider,
        verified_steps,  # type: ignore[arg-type]
    )

    assert "trace.invalid_verified_steps" in {item.code for item in result.diagnostics}
    assert provider_calls == 0


def test_public_checker_annotations_and_docs_match_exact_container_contract() -> None:
    checker_cases = (
        (check_schema_contract, "probes", SchemaProbe),
        (check_denotation_contract, "cases", DenotationCase),
        (check_state_contract, "cases", StateCase),
        (check_trace_contract, "cases", TraceCase),
        (check_trace_contract, "verified_denotation_cases", DenotationCase),
    )
    for checker, parameter, item_type in checker_cases:
        hints = get_type_hints(checker)
        assert hints[parameter] == list[item_type] | tuple[item_type, ...]
        assert "exact built-in list or tuple" in (getdoc(checker) or "")

    denotation_doc = getdoc(check_denotation_contract) or ""
    assert "twice" in denotation_doc
    assert "type-sensitive" in denotation_doc
    state_doc = getdoc(check_state_contract) or ""
    assert "opaque" in state_doc
    assert "reset" in state_doc
    trace_doc = getdoc(check_trace_contract) or ""
    assert "already-verified" in trace_doc
    assert "identity" in trace_doc
    assert "score" in trace_doc

    suite_fingerprint_hints = get_type_hints(contract_suite_fingerprint)
    assert suite_fingerprint_hints["results"] == (
        list[LayerContractResult] | tuple[LayerContractResult, ...]
    )
    assert "exact built-in list or tuple" in (getdoc(contract_suite_fingerprint) or "")

    suite_hints = get_type_hints(evaluate_contract_suite)
    assert suite_hints["schema_probes"] == list[SchemaProbe] | tuple[SchemaProbe, ...]
    assert suite_hints["denotation_cases"] == (list[DenotationCase] | tuple[DenotationCase, ...])
    assert suite_hints["state_cases"] == list[StateCase] | tuple[StateCase, ...]
    assert suite_hints["state_provider"] is state_contracts.StateEvidenceProvider
    assert suite_hints["trace_cases"] == list[TraceCase] | tuple[TraceCase, ...]
    assert suite_hints["trace_provider"] is trace_contracts.TraceEvidenceProvider
    evaluate_doc = getdoc(evaluate_contract_suite) or ""
    assert "exact built-in list or tuple" in evaluate_doc
    assert "digest" in evaluate_doc
    assert "episode" in evaluate_doc

    admission_doc = getdoc(require_dataset_admission) or ""
    assert "same process" in admission_doc
    assert "same object" in admission_doc
    assert "serialization" in admission_doc


def test_contract_internal_modules_have_the_declared_acyclic_dependencies() -> None:
    contract_root = Path(schema_contracts.__file__).parent
    allowed_dependencies = {
        "_common": set(),
        "_schema": {"_common"},
        "denotation": {"_common", "_schema"},
        "state": {"_common"},
        "trace": {"_common", "_schema", "denotation"},
        "suite": {"_common", "_schema", "denotation", "state", "trace"},
    }

    for module_name, allowed in allowed_dependencies.items():
        path = contract_root / f"{module_name}.py"
        assert path.is_file()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        dependencies: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                prefix = "toolshift.contracts."
                if node.module.startswith(prefix):
                    dependencies.add(node.module.removeprefix(prefix).split(".")[0])
            elif isinstance(node, ast.Import):
                prefix = "toolshift.contracts."
                for alias in node.names:
                    if alias.name.startswith(prefix):
                        dependencies.add(alias.name.removeprefix(prefix).split(".")[0])
        assert dependencies <= allowed
        assert "schema" not in dependencies


def test_contract_modules_import_in_any_order_with_public_alias_identity() -> None:
    orders = (
        ("schema", "suite", "denotation", "state", "trace"),
        ("trace", "state", "denotation", "suite", "schema"),
    )
    script = """
import importlib
import sys

for name in sys.argv[1:]:
    importlib.import_module(f"toolshift.contracts.{name}")

import toolshift.contracts as public
import toolshift.contracts.schema as facade
import toolshift.contracts.suite as suite

for name in (
    "ContractDiagnostic",
    "ContractSuiteResult",
    "LayerContractResult",
    "SchemaProbe",
    "check_schema_contract",
    "contract_suite_fingerprint",
    "evaluate_contract_suite",
    "require_dataset_admission",
    "schema_fingerprint",
):
    assert getattr(facade, name) is getattr(public, name)
assert facade.ContractSuiteResult is suite.ContractSuiteResult
assert facade.evaluate_contract_suite is suite.evaluate_contract_suite
assert facade.require_dataset_admission is suite.require_dataset_admission
"""
    for order in orders:
        subprocess.run(
            [sys.executable, "-c", script, *order],
            check=True,
            capture_output=True,
            text=True,
        )


def test_contract_package_imports_public_symbols_from_owning_modules() -> None:
    package_init = Path(schema_contracts.__file__).with_name("__init__.py")
    tree = ast.parse(
        package_init.read_text(encoding="utf-8"),
        filename=str(package_init),
    )
    imported_modules = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "toolshift.contracts.schema" not in imported_modules
    assert "toolshift.contracts._common" in imported_modules
    assert "toolshift.contracts._schema" in imported_modules
    assert "toolshift.contracts.suite" in imported_modules


def test_contract_internal_symbol_ownership_stays_layered() -> None:
    assert ContractDiagnostic.__module__ == "toolshift.contracts._common"
    assert LayerContractResult.__module__ == "toolshift.contracts._common"
    assert ContractSuiteResult.__module__ == "toolshift.contracts.suite"
    import toolshift.contracts._common as common_contracts

    assert not hasattr(common_contracts, "_raw_adapter_variant")
    assert not hasattr(common_contracts, "_SUITE_ATTESTATION")
    assert not hasattr(common_contracts, "_SuiteProvenance")
    assert hasattr(schema_core_contracts, "_raw_adapter_variant")
