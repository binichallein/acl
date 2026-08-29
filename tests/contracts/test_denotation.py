"""Denotation behavioral-equivalence contract tests."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from toolshift.contracts.denotation import DenotationCase, check_denotation_contract
from toolshift.types import (
    JSONValue,
    SemanticAction,
)

from ._support import (
    _ActualPipelineAdapter,
    _BadObservationAdapter,
    _EmptyCompileAdapter,
    _ExplodingDenotationAdapter,
    _IdentityAdapter,
    _NondeterministicAdapter,
    _NondeterministicCompileAdapter,
    _NoOpAdapter,
    _OneParseFailureAdapter,
    _SecondCompileDriftAdapter,
    _surface_call,
    _TwoCallAdapter,
    _variant,
)


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
    assert adapter.wrapped_actions[0] is adapter.parsed_runs[0]
    assert adapter.wrapped_actions[1] is adapter.parsed_runs[1]
    assert adapter.wrapped_actions[0][0] is adapter.parsed_actions[0]
    assert adapter.wrapped_actions[1][0] is adapter.parsed_actions[1]
    assert adapter.events == [
        "parse",
        "parse",
        "compile",
        "compile",
        "compile",
        "compile",
        "wrap",
        "wrap",
    ]
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
    assert "denotation.nondeterministic_compile" in {item.code for item in result.diagnostics}


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
    compile_result = check_denotation_contract(_NondeterministicCompileAdapter(variant), [case])
    empty_result = check_denotation_contract(_EmptyCompileAdapter(variant), [case])

    assert "denotation.nondeterministic_parse" in {item.code for item in parse_result.diagnostics}
    assert "denotation.nondeterministic_compile" in {
        item.code for item in compile_result.diagnostics
    }
    assert "denotation.compile_exception" in {item.code for item in empty_result.diagnostics}


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

    assert "denotation.compile_mismatch" in {item.code for item in compile_result.diagnostics}
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

    assert "denotation.observation_cardinality" in {item.code for item in result.diagnostics}


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
