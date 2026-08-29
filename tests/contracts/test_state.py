"""State behavioral-equivalence contract tests."""

from __future__ import annotations

import pytest

from toolshift.contracts.state import StateCase, StateEvidence, check_state_contract

from ._support import (
    _state_evidence,
)


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

    assert "state.provider_exception" in {item.code for item in evidence_result.diagnostics}
    assert "state.invalid_case" in {item.code for item in case_result.diagnostics}
