"""Single-step behavioral denotation contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.schema import (
    LayerContractResult,
    _actions_equal,
    _diagnostic,
    _fingerprint_parts,
    _make_layer_result,
    _snapshot_actions,
    _snapshot_call,
    schema_fingerprint,
)
from toolshift.types import JSONValue, SemanticAction, _freeze_json_root, canonical_json_bytes

_BaseCallGroup: TypeAlias = tuple[Mapping[str, JSONValue], ...]
_CompileRun: TypeAlias = tuple[tuple[_BaseCallGroup, ...], ...]


def _snapshot_call_groups(
    value: object, context: str
) -> tuple[tuple[Mapping[str, JSONValue], ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list or tuple")
    groups: list[tuple[Mapping[str, JSONValue], ...]] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, (list, tuple)):
            raise ValueError(f"{context} groups must be lists or tuples")
        groups.append(
            tuple(
                _snapshot_call(call, f"{context}[{group_index}]") for call in group
            )
        )
    return tuple(groups)


def _snapshot_observation_groups(
    value: object, context: str
) -> tuple[tuple[JSONValue, ...], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list or tuple")
    groups: list[tuple[JSONValue, ...]] = []
    for group_index, group in enumerate(value):
        if not isinstance(group, (list, tuple)):
            raise ValueError(f"{context} groups must be lists or tuples")
        groups.append(
            tuple(
                _freeze_json_root(item, f"{context}[{group_index}][{item_index}]")
                for item_index, item in enumerate(group)
            )
        )
    return tuple(groups)


def _json_equal(first: JSONValue, second: JSONValue) -> bool:
    return canonical_json_bytes(first) == canonical_json_bytes(second)


def _call_groups_equal(
    first: tuple[Mapping[str, JSONValue], ...],
    second: tuple[Mapping[str, JSONValue], ...],
) -> bool:
    return len(first) == len(second) and all(
        canonical_json_bytes(left) == canonical_json_bytes(right)
        for left, right in zip(first, second, strict=True)
    )


def _compile_runs_equal(first: _CompileRun, second: _CompileRun) -> bool:
    return len(first) == len(second) and all(
        len(left_trials) == len(right_trials)
        and all(
            _call_groups_equal(left_group, right_group)
            for left_group, right_group in zip(
                left_trials,
                right_trials,
                strict=True,
            )
        )
        for left_trials, right_trials in zip(first, second, strict=True)
    )


@dataclass(frozen=True, slots=True, eq=False)
class DenotationCase:
    """Expected parse, compilation, and observation wrapping for one surface call."""

    case_id: str
    surface_call: Mapping[str, JSONValue]
    expected_actions: tuple[SemanticAction, ...]
    expected_base_call_groups: tuple[tuple[Mapping[str, JSONValue], ...], ...]
    base_observation_groups: tuple[tuple[JSONValue, ...], ...]
    expected_surface_observation: JSONValue
    _snapshot_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from toolshift.contracts.schema import _require_safe_identifier

        object.__setattr__(
            self,
            "case_id",
            _require_safe_identifier(self.case_id, "case_id"),
        )
        surface_call = _snapshot_call(self.surface_call, "surface_call")
        expected_actions = _snapshot_actions(self.expected_actions, "expected_actions")
        base_call_groups = _snapshot_call_groups(
            self.expected_base_call_groups, "expected_base_call_groups"
        )
        observation_groups = _snapshot_observation_groups(
            self.base_observation_groups, "base_observation_groups"
        )
        expected_observation = _freeze_json_root(
            self.expected_surface_observation, "expected_surface_observation"
        )
        object.__setattr__(self, "surface_call", surface_call)
        object.__setattr__(self, "expected_actions", expected_actions)
        object.__setattr__(self, "expected_base_call_groups", base_call_groups)
        object.__setattr__(self, "base_observation_groups", observation_groups)
        object.__setattr__(
            self,
            "expected_surface_observation",
            expected_observation,
        )
        parts = [canonical_json_bytes(self.case_id), canonical_json_bytes(surface_call)]
        for action in expected_actions:
            parts.extend(
                (canonical_json_bytes(action.name), canonical_json_bytes(action.arguments))
            )
        parts.append(canonical_json_bytes(len(base_call_groups)))
        for group in base_call_groups:
            parts.append(canonical_json_bytes(len(group)))
            parts.extend(canonical_json_bytes(call) for call in group)
        parts.append(canonical_json_bytes(len(observation_groups)))
        for group in observation_groups:
            parts.append(canonical_json_bytes(len(group)))
            parts.extend(canonical_json_bytes(observation) for observation in group)
        parts.append(canonical_json_bytes(expected_observation))
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            _fingerprint_parts(b"toolshift.denotation-case.v1", parts),
        )


def _validate_denotation_case(value: object) -> DenotationCase:
    if type(value) is not DenotationCase:
        raise ValueError("cases must contain only DenotationCase values")
    case = cast(DenotationCase, value)
    rebuilt = DenotationCase(
        case.case_id,
        case.surface_call,
        case.expected_actions,
        case.expected_base_call_groups,
        case.base_observation_groups,
        case.expected_surface_observation,
    )
    if case._snapshot_fingerprint != rebuilt._snapshot_fingerprint:
        raise ValueError("denotation case has been mutated")
    return case


def check_denotation_contract(
    adapter: SemanticAdapter,
    cases: Sequence[DenotationCase],
) -> LayerContractResult:
    """Verify deterministic parse, compilation, cardinality, and observation wrapping."""

    diagnostics = []
    try:
        adapter_variant_fingerprint = schema_fingerprint(adapter.variant)
    except Exception:
        adapter_variant_fingerprint = None
        diagnostics.append(
            _diagnostic("denotation.invalid_adapter", "Adapter validation failed")
        )

    def adapter_binding_matches() -> bool:
        if adapter_variant_fingerprint is None:
            return False
        try:
            return schema_fingerprint(adapter.variant) == adapter_variant_fingerprint
        except Exception:
            return False

    if not isinstance(cases, (list, tuple)):
        diagnostics.append(
            _diagnostic("denotation.invalid_cases", "Denotation cases must be a finite sequence")
        )
        raw_cases: tuple[object, ...] = ()
    else:
        raw_cases = tuple(cases)
    if not raw_cases:
        diagnostics.append(
            _diagnostic("denotation.empty_cases", "Denotation cases must not be empty")
        )

    valid_cases: list[DenotationCase] = []
    for raw_case in raw_cases:
        try:
            valid_cases.append(_validate_denotation_case(raw_case))
        except Exception:
            case_id = getattr(raw_case, "case_id", "suite")
            try:
                from toolshift.contracts.schema import _require_safe_identifier

                case_id = _require_safe_identifier(case_id, "case_id")
            except ValueError:
                case_id = "suite"
            diagnostics.append(
                _diagnostic(
                    "denotation.invalid_case",
                    "Denotation case validation failed",
                    case_id,
                )
            )

    case_ids = [case.case_id for case in valid_cases]
    if len(case_ids) != len(set(case_ids)):
        diagnostics.append(
            _diagnostic(
                "denotation.duplicate_case",
                "Denotation case identifiers must be unique",
            )
        )

    for case in valid_cases:
        variant_rebound = False
        action_count = len(case.expected_actions)
        if len(case.expected_base_call_groups) != action_count:
            diagnostics.append(
                _diagnostic(
                    "denotation.base_group_count",
                    "Base call group count must equal action count",
                    case.case_id,
                )
            )
        if len(case.base_observation_groups) != action_count:
            diagnostics.append(
                _diagnostic(
                    "denotation.observation_group_count",
                    "Observation group count must equal action count",
                    case.case_id,
                )
            )

        parsed: list[tuple[SemanticAction, ...]] = []
        parsed_snapshots: list[tuple[SemanticAction, ...]] = []
        parse_raised = False
        for _ in range(2):
            try:
                output = adapter.surface_to_semantic(case.surface_call)
                if not isinstance(output, tuple):
                    raise ValueError("parse output must be a tuple")
                parsed_snapshots.append(_snapshot_actions(output, "parsed actions"))
                parsed.append(output)
            except Exception:
                parse_raised = True
            finally:
                if not adapter_binding_matches():
                    variant_rebound = True
        if parse_raised or len(parsed) != 2:
            diagnostics.append(
                _diagnostic(
                    "denotation.parse_exception",
                    "Surface parsing raised or returned malformed actions",
                    case.case_id,
                )
            )
        else:
            if not _actions_equal(parsed_snapshots[0], parsed_snapshots[1]):
                diagnostics.append(
                    _diagnostic(
                        "denotation.nondeterministic_parse",
                        "Surface parsing is not deterministic",
                        case.case_id,
                    )
                )
            if any(
                not _actions_equal(run, case.expected_actions)
                for run in parsed_snapshots
            ):
                diagnostics.append(
                    _diagnostic(
                        "denotation.parse_mismatch",
                        "Surface parsing differs from expected actions",
                        case.case_id,
                    )
                )

        compiled_runs: list[_CompileRun] = []
        compile_raised = False
        if len(parsed) == 2:
            for parsed_run in parsed:
                compiled_actions: list[tuple[_BaseCallGroup, ...]] = []
                run_raised = False
                for action in parsed_run:
                    trials: list[_BaseCallGroup] = []
                    for _ in range(2):
                        try:
                            output = adapter.semantic_to_base_calls(action)
                            if not isinstance(output, tuple):
                                raise ValueError("compile output must be a tuple")
                            group = tuple(
                                _snapshot_call(call, "compiled base calls")
                                for call in output
                            )
                            if not group:
                                raise ValueError("compile output must not be empty")
                            trials.append(group)
                        except Exception:
                            compile_raised = True
                            run_raised = True
                        finally:
                            if not adapter_binding_matches():
                                variant_rebound = True
                    if len(trials) == 2:
                        compiled_actions.append(tuple(trials))
                if not run_raised:
                    compiled_runs.append(tuple(compiled_actions))
        if compile_raised:
            diagnostics.append(
                _diagnostic(
                    "denotation.compile_exception",
                    "Action compilation raised or returned an empty malformed group",
                    case.case_id,
                )
            )

        if len(compiled_runs) == 2:
            if not _compile_runs_equal(
                compiled_runs[0], compiled_runs[1]
            ) or any(
                any(
                    not _call_groups_equal(trials[0], trial)
                    for trial in trials[1:]
                )
                for compiled_run in compiled_runs
                for trials in compiled_run
            ):
                diagnostics.append(
                    _diagnostic(
                        "denotation.nondeterministic_compile",
                        "Action compilation is not deterministic",
                        case.case_id,
                    )
                )
            if any(
                len(compiled_run) != len(case.expected_base_call_groups)
                or any(
                    any(
                        not _call_groups_equal(group, expected_group)
                        for group in trials
                    )
                    for trials, expected_group in zip(
                        compiled_run,
                        case.expected_base_call_groups,
                        strict=True,
                    )
                )
                for compiled_run in compiled_runs
            ):
                diagnostics.append(
                    _diagnostic(
                        "denotation.compile_mismatch",
                        "Action compilation differs from expected base calls",
                        case.case_id,
                    )
                )

            if any(
                len(compiled_run) != len(case.base_observation_groups)
                or any(
                    any(
                        len(observation_group) != len(compiled_group)
                        for compiled_group in trials
                    )
                    for observation_group, trials in zip(
                        case.base_observation_groups,
                        compiled_run,
                        strict=True,
                    )
                )
                for compiled_run in compiled_runs
            ):
                diagnostics.append(
                    _diagnostic(
                        "denotation.observation_cardinality",
                        "Observation cardinality must equal physical call count",
                        case.case_id,
                    )
                )

        wrapped_runs: list[JSONValue] = []
        observation_raised = False
        if len(parsed) == 2:
            for parsed_run in parsed:
                try:
                    wrapped_runs.append(
                        _freeze_json_root(
                            adapter.base_observation_to_surface(
                                case.surface_call,
                                parsed_run,
                                case.base_observation_groups,
                            ),
                            "wrapped surface observation",
                        )
                    )
                except Exception:
                    observation_raised = True
                finally:
                    if not adapter_binding_matches():
                        variant_rebound = True
        if observation_raised:
            diagnostics.append(
                _diagnostic(
                    "denotation.observation_exception",
                    "Observation wrapping raised or returned malformed JSON",
                    case.case_id,
                )
            )
        if len(wrapped_runs) == 2:
            if not _json_equal(wrapped_runs[0], wrapped_runs[1]):
                diagnostics.append(
                    _diagnostic(
                        "denotation.nondeterministic_observation",
                        "Observation wrapping is not deterministic",
                        case.case_id,
                    )
                )
            if any(
                not _json_equal(wrapped, case.expected_surface_observation)
                for wrapped in wrapped_runs
            ):
                diagnostics.append(
                    _diagnostic(
                        "denotation.observation_mismatch",
                        "Wrapped observation differs from expected observation",
                        case.case_id,
                    )
                )
        if variant_rebound:
            diagnostics.append(
                _diagnostic(
                    "denotation.variant_rebound",
                    "Adapter variant changed during denotation checks",
                    case.case_id,
                )
            )

    fingerprint = _fingerprint_parts(
        b"toolshift.denotation-suite.v1",
        [canonical_json_bytes(case.case_id) for case in valid_cases]
        + [bytes.fromhex(case._snapshot_fingerprint) for case in valid_cases],
    )
    return _make_layer_result(
        "denotation",
        fingerprint,
        len(valid_cases),
        diagnostics,
    )


__all__ = ["DenotationCase", "check_denotation_contract"]
