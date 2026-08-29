"""Episode trace, physical-effect, and score behavioral contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.denotation import DenotationCase, _validate_denotation_case
from toolshift.contracts.schema import (
    LayerContractResult,
    _actions_equal,
    _diagnostic,
    _fingerprint_parts,
    _make_layer_result,
    _require_safe_identifier,
    _require_sha256,
    _snapshot_actions,
    schema_fingerprint,
)
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SemanticAction,
    _freeze_json_root,
    canonical_json_bytes,
)


def base_call_fingerprint(call: Mapping[str, JSONValue]) -> str:
    """Return the contract fingerprint for one physical base call."""

    return hashlib.sha256(canonical_json_bytes(call)).hexdigest()


def _snapshot_trace(value: object, context: str) -> ExecutionTrace:
    if type(value) is not ExecutionTrace:
        raise ValueError(f"{context} must be an ExecutionTrace")
    trace = cast(ExecutionTrace, value)
    actions = _snapshot_actions(trace.semantic_actions, f"{context} semantic actions")
    rebuilt = ExecutionTrace(trace.surface_calls, actions, trace.base_calls)
    if trace != rebuilt:
        raise ValueError(f"{context} has been mutated")
    return rebuilt


def _call_sequences_equal(
    first: tuple[Mapping[str, JSONValue], ...],
    second: tuple[Mapping[str, JSONValue], ...],
) -> bool:
    return len(first) == len(second) and all(
        canonical_json_bytes(left) == canonical_json_bytes(right)
        for left, right in zip(first, second, strict=True)
    )


@dataclass(frozen=True, slots=True, eq=False)
class PhysicalCallEffect:
    """Opaque effect evidence aligned to one indexed physical base call."""

    call_index: int
    base_call_fingerprint: str
    effect_digest: str
    _snapshot_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.call_index, bool) or not isinstance(self.call_index, int):
            raise ValueError("call_index must be a nonnegative integer")
        if self.call_index < 0:
            raise ValueError("call_index must be a nonnegative integer")
        object.__setattr__(
            self,
            "base_call_fingerprint",
            _require_sha256(self.base_call_fingerprint, "base_call_fingerprint"),
        )
        object.__setattr__(
            self,
            "effect_digest",
            _require_safe_identifier(self.effect_digest, "effect_digest"),
        )
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            _fingerprint_parts(
                b"toolshift.physical-effect.v1",
                [
                    canonical_json_bytes(self.call_index),
                    canonical_json_bytes(self.base_call_fingerprint),
                    canonical_json_bytes(self.effect_digest),
                ],
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class TraceCase:
    """Safe identifiers for one paired reference and candidate trace."""

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
                b"toolshift.trace-case.v1",
                [
                    canonical_json_bytes(self.case_id),
                    canonical_json_bytes(self.episode_id),
                ],
            ),
        )


def _snapshot_effect(value: object, context: str) -> PhysicalCallEffect:
    if type(value) is not PhysicalCallEffect:
        raise ValueError(f"{context} must contain only PhysicalCallEffect values")
    effect = cast(PhysicalCallEffect, value)
    rebuilt = PhysicalCallEffect(
        effect.call_index,
        effect.base_call_fingerprint,
        effect.effect_digest,
    )
    if effect._snapshot_fingerprint != rebuilt._snapshot_fingerprint:
        raise ValueError(f"{context} contains mutated effect evidence")
    return rebuilt


@dataclass(frozen=True, slots=True, eq=False)
class TraceEvidence:
    """Provider evidence for reconstructed trace, score, and physical effects."""

    reference_trace: ExecutionTrace
    candidate_trace: ExecutionTrace
    candidate_steps: tuple[DenotationCase, ...]
    reference_score: JSONValue
    candidate_score: JSONValue
    reference_effects: tuple[PhysicalCallEffect, ...]
    candidate_effects: tuple[PhysicalCallEffect, ...]
    _snapshot_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        reference_trace = _snapshot_trace(self.reference_trace, "reference_trace")
        candidate_trace = _snapshot_trace(self.candidate_trace, "candidate_trace")
        if not isinstance(self.candidate_steps, (list, tuple)):
            raise ValueError("candidate_steps must be a list or tuple")
        candidate_steps = tuple(
            _validate_denotation_case(step) for step in self.candidate_steps
        )
        reference_score = _freeze_json_root(self.reference_score, "reference_score")
        candidate_score = _freeze_json_root(self.candidate_score, "candidate_score")
        if not isinstance(self.reference_effects, (list, tuple)):
            raise ValueError("reference_effects must be a list or tuple")
        if not isinstance(self.candidate_effects, (list, tuple)):
            raise ValueError("candidate_effects must be a list or tuple")
        reference_effects = tuple(
            _snapshot_effect(effect, "reference_effects")
            for effect in self.reference_effects
        )
        candidate_effects = tuple(
            _snapshot_effect(effect, "candidate_effects")
            for effect in self.candidate_effects
        )
        object.__setattr__(self, "reference_trace", reference_trace)
        object.__setattr__(self, "candidate_trace", candidate_trace)
        object.__setattr__(self, "candidate_steps", candidate_steps)
        object.__setattr__(self, "reference_score", reference_score)
        object.__setattr__(self, "candidate_score", candidate_score)
        object.__setattr__(self, "reference_effects", reference_effects)
        object.__setattr__(self, "candidate_effects", candidate_effects)

        parts = [
            canonical_json_bytes(reference_trace.surface_calls),
            canonical_json_bytes(reference_trace.base_calls),
            canonical_json_bytes(candidate_trace.surface_calls),
            canonical_json_bytes(candidate_trace.base_calls),
            canonical_json_bytes(reference_score),
            canonical_json_bytes(candidate_score),
        ]
        for trace in (reference_trace, candidate_trace):
            parts.append(canonical_json_bytes(len(trace.semantic_actions)))
            for action in trace.semantic_actions:
                parts.extend(
                    (canonical_json_bytes(action.name), canonical_json_bytes(action.arguments))
                )
        parts.extend(bytes.fromhex(step._snapshot_fingerprint) for step in candidate_steps)
        parts.extend(
            bytes.fromhex(effect._snapshot_fingerprint) for effect in reference_effects
        )
        parts.extend(
            bytes.fromhex(effect._snapshot_fingerprint) for effect in candidate_effects
        )
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            _fingerprint_parts(b"toolshift.trace-evidence.v1", parts),
        )


class TraceEvidenceProvider(Protocol):
    """Obtain trace, score, and physical-effect evidence for one case."""

    def __call__(self, case: TraceCase) -> TraceEvidence:
        """Return complete paired trace evidence."""

        ...


def _validate_trace_case(value: object) -> TraceCase:
    if type(value) is not TraceCase:
        raise ValueError("cases must contain only TraceCase values")
    case = cast(TraceCase, value)
    rebuilt = TraceCase(case.case_id, case.episode_id)
    if case._snapshot_fingerprint != rebuilt._snapshot_fingerprint:
        raise ValueError("trace case has been mutated")
    return case


def _validate_trace_evidence(value: object) -> TraceEvidence:
    if type(value) is not TraceEvidence:
        raise ValueError("provider must return TraceEvidence")
    evidence = cast(TraceEvidence, value)
    rebuilt = TraceEvidence(
        evidence.reference_trace,
        evidence.candidate_trace,
        evidence.candidate_steps,
        evidence.reference_score,
        evidence.candidate_score,
        evidence.reference_effects,
        evidence.candidate_effects,
    )
    if evidence._snapshot_fingerprint != rebuilt._snapshot_fingerprint:
        raise ValueError("trace evidence has been mutated")
    return evidence


def _validate_step_structure(step: DenotationCase) -> bool:
    if len(step.expected_actions) != len(step.expected_base_call_groups):
        return False
    if len(step.expected_actions) != len(step.base_observation_groups):
        return False
    return all(group for group in step.expected_base_call_groups) and all(
        len(observations) == len(calls)
        for calls, observations in zip(
            step.expected_base_call_groups,
            step.base_observation_groups,
            strict=True,
        )
    )


def _check_effect_alignment(
    *,
    side: str,
    case_id: str,
    base_calls: tuple[Mapping[str, JSONValue], ...],
    effects: tuple[PhysicalCallEffect, ...],
    diagnostics: list,
) -> None:
    if len(effects) != len(base_calls):
        diagnostics.append(
            _diagnostic(
                f"trace.{side}_effect_count",
                f"{side.title()} effect count must equal physical call count",
                case_id,
            )
        )
    for index, (call, effect) in enumerate(zip(base_calls, effects, strict=False)):
        if effect.call_index != index or effect.base_call_fingerprint != base_call_fingerprint(
            call
        ):
            diagnostics.append(
                _diagnostic(
                    f"trace.{side}_effect_alignment",
                    f"{side.title()} effects must align with indexed base calls",
                    case_id,
                )
            )


def check_trace_contract(
    adapter: SemanticAdapter,
    cases: Sequence[TraceCase],
    provider: TraceEvidenceProvider,
    verified_denotation_cases: Sequence[DenotationCase],
) -> LayerContractResult:
    """Verify reconstructed traces without rerunning per-step adapter mappings."""

    diagnostics = []
    try:
        adapter_variant_fingerprint = schema_fingerprint(adapter.variant)
    except Exception:
        adapter_variant_fingerprint = None
        diagnostics.append(_diagnostic("trace.invalid_adapter", "Adapter validation failed"))

    def adapter_binding_matches() -> bool:
        if adapter_variant_fingerprint is None:
            return False
        try:
            return schema_fingerprint(adapter.variant) == adapter_variant_fingerprint
        except Exception:
            return False

    if not isinstance(cases, (list, tuple)):
        diagnostics.append(
            _diagnostic("trace.invalid_cases", "Trace cases must be a finite sequence")
        )
        raw_cases: tuple[object, ...] = ()
    else:
        raw_cases = tuple(cases)
    if not raw_cases:
        diagnostics.append(_diagnostic("trace.empty_cases", "Trace cases must not be empty"))

    valid_cases: list[TraceCase] = []
    for raw_case in raw_cases:
        try:
            valid_cases.append(_validate_trace_case(raw_case))
        except Exception:
            case_id = getattr(raw_case, "case_id", "suite")
            try:
                case_id = _require_safe_identifier(case_id, "case_id")
            except ValueError:
                case_id = "suite"
            diagnostics.append(
                _diagnostic("trace.invalid_case", "Trace case validation failed", case_id)
            )

    case_ids = [case.case_id for case in valid_cases]
    episode_ids = [case.episode_id for case in valid_cases]
    if len(case_ids) != len(set(case_ids)):
        diagnostics.append(
            _diagnostic("trace.duplicate_case", "Trace case identifiers must be unique")
        )
    if len(episode_ids) != len(set(episode_ids)):
        diagnostics.append(
            _diagnostic("trace.duplicate_episode", "Trace episode identifiers must be unique")
        )

    verified_by_id: dict[str, DenotationCase] = {}
    if not isinstance(verified_denotation_cases, (list, tuple)):
        diagnostics.append(
            _diagnostic(
                "trace.invalid_verified_steps",
                "Verified denotation cases must be a finite sequence",
            )
        )
    else:
        for raw_step in verified_denotation_cases:
            try:
                step = _validate_denotation_case(raw_step)
            except Exception:
                diagnostics.append(
                    _diagnostic(
                        "trace.invalid_verified_step",
                        "Verified denotation case validation failed",
                    )
                )
                continue
            if step.case_id in verified_by_id:
                diagnostics.append(
                    _diagnostic(
                        "trace.duplicate_verified_step",
                        "Verified denotation identifiers must be unique",
                    )
                )
            else:
                verified_by_id[step.case_id] = step

    fingerprint_parts = [
        bytes.fromhex(case._snapshot_fingerprint) for case in valid_cases
    ]
    surface_steps_run = 0
    for case in valid_cases:
        try:
            evidence = _validate_trace_evidence(provider(case))
        except Exception:
            diagnostics.append(
                _diagnostic(
                    "trace.provider_exception",
                    "Trace evidence provider raised or returned malformed evidence",
                    case.case_id,
                )
            )
            fingerprint_parts.append(b"provider-error")
            continue
        fingerprint_parts.append(bytes.fromhex(evidence._snapshot_fingerprint))
        surface_steps_run += len(evidence.candidate_steps)
        if not evidence.candidate_steps:
            diagnostics.append(
                _diagnostic(
                    "trace.empty_steps",
                    "Candidate trace must contain a verified surface step",
                    case.case_id,
                )
            )

        reconstructed_surface: list[Mapping[str, JSONValue]] = []
        reconstructed_actions: list[SemanticAction] = []
        reconstructed_base: list[Mapping[str, JSONValue]] = []
        for step in evidence.candidate_steps:
            registered = verified_by_id.get(step.case_id)
            if registered is not step:
                diagnostics.append(
                    _diagnostic(
                        "trace.unverified_step",
                        "Candidate trace contains an unverified denotation object",
                        case.case_id,
                    )
                )
            if not _validate_step_structure(step):
                diagnostics.append(
                    _diagnostic(
                        "trace.invalid_step_structure",
                        "Candidate denotation step has invalid group structure",
                        case.case_id,
                    )
                )
            reconstructed_surface.append(step.surface_call)
            reconstructed_actions.extend(step.expected_actions)
            for group in step.expected_base_call_groups:
                reconstructed_base.extend(group)

        reconstructed = ExecutionTrace(
            reconstructed_surface,
            reconstructed_actions,
            reconstructed_base,
        )
        if not _call_sequences_equal(
            reconstructed.surface_calls, evidence.candidate_trace.surface_calls
        ):
            diagnostics.append(
                _diagnostic(
                    "trace.surface_reconstruction_mismatch",
                    "Reconstructed surface calls differ from candidate trace",
                    case.case_id,
                )
            )
        if not _actions_equal(
            reconstructed.semantic_actions, evidence.candidate_trace.semantic_actions
        ):
            diagnostics.append(
                _diagnostic(
                    "trace.semantic_reconstruction_mismatch",
                    "Reconstructed semantic actions differ from candidate trace",
                    case.case_id,
                )
            )
        if not _call_sequences_equal(
            reconstructed.base_calls, evidence.candidate_trace.base_calls
        ):
            diagnostics.append(
                _diagnostic(
                    "trace.base_reconstruction_mismatch",
                    "Reconstructed base calls differ from candidate trace",
                    case.case_id,
                )
            )

        variant_rebound = False
        try:
            try:
                reference_semantics = adapter.canonicalize_trace(
                    evidence.reference_trace
                )
            finally:
                if not adapter_binding_matches():
                    variant_rebound = True
            try:
                candidate_semantics = adapter.canonicalize_trace(
                    evidence.candidate_trace
                )
            finally:
                if not adapter_binding_matches():
                    variant_rebound = True
            if not isinstance(reference_semantics, tuple) or not isinstance(
                candidate_semantics, tuple
            ):
                raise ValueError("canonical semantics must be tuples")
            reference_semantics = _snapshot_actions(
                reference_semantics, "reference canonical semantics"
            )
            candidate_semantics = _snapshot_actions(
                candidate_semantics, "candidate canonical semantics"
            )
            if (
                not _actions_equal(candidate_semantics, reconstructed.semantic_actions)
                or not _actions_equal(
                    candidate_semantics,
                    evidence.candidate_trace.semantic_actions,
                )
                or not _actions_equal(
                    reference_semantics,
                    evidence.reference_trace.semantic_actions,
                )
            ):
                diagnostics.append(
                    _diagnostic(
                        "trace.canonical_semantic_channel_mismatch",
                        "Candidate canonical semantics differ from reconstructed actions",
                        case.case_id,
                    )
                )
            if not _actions_equal(reference_semantics, candidate_semantics):
                diagnostics.append(
                    _diagnostic(
                        "trace.semantic_mismatch",
                        "Reference and candidate canonical semantics differ",
                        case.case_id,
                    )
                )
        except Exception:
            diagnostics.append(
                _diagnostic(
                    "trace.canonicalization_exception",
                    "Trace canonicalization raised or returned malformed actions",
                    case.case_id,
                )
            )
        if variant_rebound:
            diagnostics.append(
                _diagnostic(
                    "trace.variant_rebound",
                    "Adapter variant changed during trace canonicalization",
                    case.case_id,
                )
            )

        if not _call_sequences_equal(
            evidence.reference_trace.base_calls,
            evidence.candidate_trace.base_calls,
        ):
            diagnostics.append(
                _diagnostic(
                    "trace.base_call_mismatch",
                    "Reference and candidate base call order differs",
                    case.case_id,
                )
            )

        _check_effect_alignment(
            side="reference",
            case_id=case.case_id,
            base_calls=evidence.reference_trace.base_calls,
            effects=evidence.reference_effects,
            diagnostics=diagnostics,
        )
        _check_effect_alignment(
            side="candidate",
            case_id=case.case_id,
            base_calls=evidence.candidate_trace.base_calls,
            effects=evidence.candidate_effects,
            diagnostics=diagnostics,
        )
        if len(evidence.reference_effects) == len(evidence.candidate_effects) and any(
            (
                reference.call_index != candidate.call_index
                or reference.base_call_fingerprint != candidate.base_call_fingerprint
                or reference.effect_digest != candidate.effect_digest
            )
            for reference, candidate in zip(
                evidence.reference_effects,
                evidence.candidate_effects,
                strict=True,
            )
        ):
            diagnostics.append(
                _diagnostic(
                    "trace.effect_mismatch",
                    "Reference and candidate physical effects differ",
                    case.case_id,
                )
            )

        if canonical_json_bytes(evidence.reference_score) != canonical_json_bytes(
            evidence.candidate_score
        ):
            diagnostics.append(
                _diagnostic(
                    "trace.score_mismatch",
                    "Reference and candidate scores differ",
                    case.case_id,
                )
            )

    return _make_layer_result(
        "trace",
        _fingerprint_parts(b"toolshift.trace-suite.v1", fingerprint_parts),
        surface_steps_run,
        diagnostics,
    )


__all__ = [
    "PhysicalCallEffect",
    "TraceCase",
    "TraceEvidence",
    "TraceEvidenceProvider",
    "base_call_fingerprint",
    "check_trace_contract",
]
