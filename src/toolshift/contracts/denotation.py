"""Single-step behavioral denotation contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.schema import (
    LayerContractResult,
    _actions_equal,
    _diagnostic,
    _fingerprint_parts,
    _make_layer_result,
    _snapshot_actions,
    _snapshot_call,
    _validate_schema_variant,
)
from toolshift.types import JSONValue, SemanticAction, _freeze_json_root, canonical_json_bytes


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
        _validate_schema_variant(adapter.variant)
    except Exception:
        diagnostics.append(
            _diagnostic("denotation.invalid_adapter", "Adapter validation failed")
        )

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
        parse_raised = False
        for _ in range(2):
            try:
                output = adapter.surface_to_semantic(case.surface_call)
                if not isinstance(output, tuple):
                    raise ValueError("parse output must be a tuple")
                parsed.append(_snapshot_actions(output, "parsed actions"))
            except Exception:
                parse_raised = True
        if parse_raised or len(parsed) != 2:
            diagnostics.append(
                _diagnostic(
                    "denotation.parse_exception",
                    "Surface parsing raised or returned malformed actions",
                    case.case_id,
                )
            )
        else:
            if not _actions_equal(parsed[0], parsed[1]):
                diagnostics.append(
                    _diagnostic(
                        "denotation.nondeterministic_parse",
                        "Surface parsing is not deterministic",
                        case.case_id,
                    )
                )
            if not _actions_equal(parsed[0], case.expected_actions):
                diagnostics.append(
                    _diagnostic(
                        "denotation.parse_mismatch",
                        "Surface parsing differs from expected actions",
                        case.case_id,
                    )
                )

        physical_counts: list[int] = []
        for action_index, action in enumerate(case.expected_actions):
            compiled: list[tuple[Mapping[str, JSONValue], ...]] = []
            compile_raised = False
            for _ in range(2):
                try:
                    output = adapter.semantic_to_base_calls(action)
                    if not isinstance(output, tuple):
                        raise ValueError("compile output must be a tuple")
                    group = tuple(
                        _snapshot_call(call, "compiled base calls") for call in output
                    )
                    if not group:
                        raise ValueError("compile output must not be empty")
                    compiled.append(group)
                except Exception:
                    compile_raised = True
            expected_group = (
                case.expected_base_call_groups[action_index]
                if action_index < len(case.expected_base_call_groups)
                else ()
            )
            physical_counts.append(len(compiled[0]) if compiled else len(expected_group))
            if compile_raised or len(compiled) != 2:
                diagnostics.append(
                    _diagnostic(
                        "denotation.compile_exception",
                        "Action compilation raised or returned an empty malformed group",
                        case.case_id,
                    )
                )
                continue
            if not _call_groups_equal(compiled[0], compiled[1]):
                diagnostics.append(
                    _diagnostic(
                        "denotation.nondeterministic_compile",
                        "Action compilation is not deterministic",
                        case.case_id,
                    )
                )
            if not _call_groups_equal(compiled[0], expected_group):
                diagnostics.append(
                    _diagnostic(
                        "denotation.compile_mismatch",
                        "Action compilation differs from expected base calls",
                        case.case_id,
                    )
                )

        for action_index, expected_count in enumerate(physical_counts):
            if (
                action_index >= len(case.base_observation_groups)
                or len(case.base_observation_groups[action_index]) != expected_count
            ):
                diagnostics.append(
                    _diagnostic(
                        "denotation.observation_cardinality",
                        "Observation cardinality must equal physical call count",
                        case.case_id,
                    )
                )

        try:
            wrapped = _freeze_json_root(
                adapter.base_observation_to_surface(
                    case.surface_call,
                    case.expected_actions,
                    case.base_observation_groups,
                ),
                "wrapped surface observation",
            )
            if not _json_equal(wrapped, case.expected_surface_observation):
                diagnostics.append(
                    _diagnostic(
                        "denotation.observation_mismatch",
                        "Wrapped observation differs from expected observation",
                        case.case_id,
                    )
                )
        except Exception:
            diagnostics.append(
                _diagnostic(
                    "denotation.observation_exception",
                    "Observation wrapping raised or returned malformed JSON",
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
