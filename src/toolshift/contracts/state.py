"""Opaque final-state and reset behavioral contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from toolshift.contracts.schema import (
    LayerContractResult,
    _diagnostic,
    _fingerprint_parts,
    _make_layer_result,
    _require_safe_identifier,
    _require_sha256,
)
from toolshift.types import canonical_json_bytes


@dataclass(frozen=True, slots=True, eq=False)
class StateCase:
    """Safe identifiers for one paired reference and candidate episode."""

    case_id: str
    episode_id: str
    _snapshot_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _require_safe_identifier(self.case_id, "case_id"),
        )
        object.__setattr__(
            self,
            "episode_id",
            _require_safe_identifier(self.episode_id, "episode_id"),
        )
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            _fingerprint_parts(
                b"toolshift.state-case.v1",
                [
                    canonical_json_bytes(self.case_id),
                    canonical_json_bytes(self.episode_id),
                ],
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class StateEvidence:
    """Provider-supplied opaque state and collateral digests."""

    reference_initial_sha256: str
    candidate_initial_sha256: str
    reference_final_sha256: str
    candidate_final_sha256: str
    reference_post_reset_sha256: str
    candidate_post_reset_sha256: str
    reference_collateral_digest: str
    candidate_collateral_digest: str
    _snapshot_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        sha_fields = (
            "reference_initial_sha256",
            "candidate_initial_sha256",
            "reference_final_sha256",
            "candidate_final_sha256",
            "reference_post_reset_sha256",
            "candidate_post_reset_sha256",
        )
        for name in sha_fields:
            object.__setattr__(self, name, _require_sha256(getattr(self, name), name))
        collateral_fields = (
            "reference_collateral_digest",
            "candidate_collateral_digest",
        )
        for name in collateral_fields:
            object.__setattr__(
                self,
                name,
                _require_safe_identifier(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            _fingerprint_parts(
                b"toolshift.state-evidence.v1",
                [
                    canonical_json_bytes(getattr(self, name))
                    for name in (*sha_fields, *collateral_fields)
                ],
            ),
        )


class StateEvidenceProvider(Protocol):
    """Obtain opaque state evidence for one case."""

    def __call__(self, case: StateCase) -> StateEvidence:
        """Return complete paired evidence without exposing state payloads."""

        ...


def _validate_state_case(value: object) -> StateCase:
    if type(value) is not StateCase:
        raise ValueError("cases must contain only StateCase values")
    case = cast(StateCase, value)
    rebuilt = StateCase(case.case_id, case.episode_id)
    if case._snapshot_fingerprint != rebuilt._snapshot_fingerprint:
        raise ValueError("state case has been mutated")
    return case


def _validate_state_evidence(value: object) -> StateEvidence:
    if type(value) is not StateEvidence:
        raise ValueError("provider must return StateEvidence")
    evidence = cast(StateEvidence, value)
    rebuilt = StateEvidence(
        evidence.reference_initial_sha256,
        evidence.candidate_initial_sha256,
        evidence.reference_final_sha256,
        evidence.candidate_final_sha256,
        evidence.reference_post_reset_sha256,
        evidence.candidate_post_reset_sha256,
        evidence.reference_collateral_digest,
        evidence.candidate_collateral_digest,
    )
    if evidence._snapshot_fingerprint != rebuilt._snapshot_fingerprint:
        raise ValueError("state evidence has been mutated")
    return evidence


def check_state_contract(
    cases: Sequence[StateCase],
    provider: StateEvidenceProvider,
) -> LayerContractResult:
    """Verify paired start, final, collateral, and reset digests."""

    diagnostics = []
    if not isinstance(cases, (list, tuple)):
        diagnostics.append(
            _diagnostic("state.invalid_cases", "State cases must be a finite sequence")
        )
        raw_cases: tuple[object, ...] = ()
    else:
        raw_cases = tuple(cases)
    if not raw_cases:
        diagnostics.append(_diagnostic("state.empty_cases", "State cases must not be empty"))

    valid_cases: list[StateCase] = []
    for raw_case in raw_cases:
        try:
            valid_cases.append(_validate_state_case(raw_case))
        except Exception:
            case_id = getattr(raw_case, "case_id", "suite")
            try:
                case_id = _require_safe_identifier(case_id, "case_id")
            except ValueError:
                case_id = "suite"
            diagnostics.append(
                _diagnostic("state.invalid_case", "State case validation failed", case_id)
            )

    case_ids = [case.case_id for case in valid_cases]
    episode_ids = [case.episode_id for case in valid_cases]
    if len(case_ids) != len(set(case_ids)):
        diagnostics.append(
            _diagnostic("state.duplicate_case", "State case identifiers must be unique")
        )
    if len(episode_ids) != len(set(episode_ids)):
        diagnostics.append(
            _diagnostic("state.duplicate_episode", "State episode identifiers must be unique")
        )

    fingerprint_parts = [
        bytes.fromhex(case._snapshot_fingerprint) for case in valid_cases
    ]
    for case in valid_cases:
        try:
            evidence = _validate_state_evidence(provider(case))
        except Exception:
            diagnostics.append(
                _diagnostic(
                    "state.provider_exception",
                    "State evidence provider raised or returned malformed evidence",
                    case.case_id,
                )
            )
            fingerprint_parts.append(b"provider-error")
            continue

        fingerprint_parts.append(bytes.fromhex(evidence._snapshot_fingerprint))
        if evidence.reference_initial_sha256 != evidence.candidate_initial_sha256:
            diagnostics.append(
                _diagnostic(
                    "state.initial_mismatch",
                    "Reference and candidate initial states differ",
                    case.case_id,
                )
            )
        if evidence.reference_final_sha256 != evidence.candidate_final_sha256:
            diagnostics.append(
                _diagnostic(
                    "state.final_mismatch",
                    "Reference and candidate final states differ",
                    case.case_id,
                )
            )
        if evidence.reference_collateral_digest != evidence.candidate_collateral_digest:
            diagnostics.append(
                _diagnostic(
                    "state.collateral_mismatch",
                    "Reference and candidate collateral digests differ",
                    case.case_id,
                )
            )
        if evidence.reference_post_reset_sha256 != evidence.reference_initial_sha256:
            diagnostics.append(
                _diagnostic(
                    "state.reference_reset_mismatch",
                    "Reference reset did not restore its initial state",
                    case.case_id,
                )
            )
        if evidence.candidate_post_reset_sha256 != evidence.candidate_initial_sha256:
            diagnostics.append(
                _diagnostic(
                    "state.candidate_reset_mismatch",
                    "Candidate reset did not restore its initial state",
                    case.case_id,
                )
            )

    return _make_layer_result(
        "state",
        _fingerprint_parts(b"toolshift.state-suite.v1", fingerprint_parts),
        len(valid_cases),
        diagnostics,
    )


__all__ = [
    "StateCase",
    "StateEvidence",
    "StateEvidenceProvider",
    "check_state_contract",
]
