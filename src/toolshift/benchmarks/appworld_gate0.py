"""Process-local paired evidence for the AppWorld Gate 0 smoke.

This module owns no AppWorld lifecycle globally and deliberately persists no
contract suite, episode record, task identity, call, observation, or digest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MethodType
from typing import Protocol

from toolshift.adapters import AppWorldSemanticAdapter, build_minimal_source_calls
from toolshift.adapters.semantic import SemanticAdapter
from toolshift.benchmarks.appworld_replay import canonical_state_sha256
from toolshift.benchmarks.appworld_runtime import (
    AppWorldEpisodeExecutor,
    _CapturedOraclePlan,
)
from toolshift.contracts import (
    DenotationCase,
    SchemaProbe,
    StateCase,
    StateEvidence,
    TraceCase,
    TraceEvidence,
    evaluate_contract_suite,
    require_dataset_admission,
)
from toolshift.transforms import ParameterRestructureAdapter, RenameAdapter
from toolshift.types import JSONValue, canonical_json_bytes

_PAIR_EVIDENCE_FAILURE = "paired AppWorld evidence failed"
_PAIR_EXECUTION_FAILURE = "paired AppWorld execution failed"
_INVALID_PAIR = "paired AppWorld evidence input is invalid"
_INVALID_PLAN = "paired admission requires a captured oracle plan"
_UNTRUSTED_TRANSLATOR = "paired admission requires a trusted canonical translator"
_SCHEMA_PROBE_FAILURE = "schema probe construction failed"
_FRESH_WORLD_FAILURE = "paired admission requires fresh world identity"
_STATE_LOOKUP_ERROR = "state evidence lookup requires the registered state case"
_TRACE_LOOKUP_ERROR = "trace evidence lookup requires the registered trace case"
_STATIC_ADMISSION_FAILURES = frozenset(
    {
        _PAIR_EVIDENCE_FAILURE,
        _PAIR_EXECUTION_FAILURE,
        _INVALID_PLAN,
        _FRESH_WORLD_FAILURE,
    }
)


class _FreshWorldFactory(Protocol):
    def __call__(self, *, role: str) -> AbstractContextManager[object]: ...


@dataclass(frozen=True, slots=True, repr=False)
class _VariantAdmissionRecord:
    admitted_suite_count: int
    checks_run: tuple[tuple[str, int], ...]
    diagnostic_codes: tuple[str, ...]


class _VariantAdmissionError(ValueError):
    """Static process-local failure carrying only payload-safe diagnostic codes."""

    def __init__(self, message: str, diagnostic_codes: tuple[str, ...]) -> None:
        safe_message = (
            message
            if type(message) is str and message in _STATIC_ADMISSION_FAILURES
            else _PAIR_EXECUTION_FAILURE
        )
        super().__init__(safe_message)
        try:
            self.diagnostic_codes = tuple(
                code
                if type(code) is str
                and 0 < len(code) <= 96
                and all(
                    character.isascii() and (character.isalnum() or character in "._-")
                    for character in code
                )
                else "pair.unsafe_diagnostic"
                for code in diagnostic_codes
            )
        except BaseException:
            self.diagnostic_codes = ("pair.unsafe_diagnostic",)

    def __repr__(self) -> str:
        return "_VariantAdmissionError(<private>)"


def _identity_canonical_call_to_surface(
    call: Mapping[str, JSONValue],
) -> Mapping[str, JSONValue]:
    return call


class _StateEvidenceLookup:
    __slots__ = ("_entries",)

    def __init__(
        self,
        entries: tuple[tuple[StateCase, StateEvidence], ...],
    ) -> None:
        if type(entries) is not tuple or any(
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not StateCase
            or type(entry[1]) is not StateEvidence
            for entry in entries
        ):
            raise ValueError(_STATE_LOOKUP_ERROR)
        self._entries = entries

    def __call__(self, case: StateCase) -> StateEvidence:
        for registered, evidence in self._entries:
            if case is registered:
                return evidence
        raise ValueError(_STATE_LOOKUP_ERROR)


class _TraceEvidenceLookup:
    __slots__ = ("_entries",)

    def __init__(
        self,
        entries: tuple[tuple[TraceCase, TraceEvidence], ...],
    ) -> None:
        if type(entries) is not tuple or any(
            type(entry) is not tuple
            or len(entry) != 2
            or type(entry[0]) is not TraceCase
            or type(entry[1]) is not TraceEvidence
            for entry in entries
        ):
            raise ValueError(_TRACE_LOOKUP_ERROR)
        self._entries = entries

    def __call__(self, case: TraceCase) -> TraceEvidence:
        for registered, evidence in self._entries:
            if case is registered:
                return evidence
        raise ValueError(_TRACE_LOOKUP_ERROR)


def _require_trusted_translator(
    source_adapter: AppWorldSemanticAdapter,
    candidate_adapter: SemanticAdapter,
    translator: object,
) -> Callable[[Mapping[str, JSONValue]], Mapping[str, JSONValue]]:
    if candidate_adapter is source_adapter:
        if translator is not _identity_canonical_call_to_surface:
            raise ValueError(_UNTRUSTED_TRANSLATOR)
        return _identity_canonical_call_to_surface
    if type(candidate_adapter) not in (RenameAdapter, ParameterRestructureAdapter):
        raise ValueError(_UNTRUSTED_TRANSLATOR)
    try:
        candidate_type = type(candidate_adapter)
        binding_validator = candidate_type.__dict__["_require_bindings"]
        binding_validator(candidate_adapter)
        if (
            object.__getattribute__(candidate_adapter, "_source_adapter") is not source_adapter
            or object.__getattribute__(candidate_adapter, "_source_adapter_seal")
            is not source_adapter
        ):
            raise ValueError
        transform = candidate_adapter.transform
        expected_function = type(transform).__dict__["canonical_call_to_surface"]
        if (
            not isinstance(translator, MethodType)
            or translator.__self__ is not transform
            or translator.__func__ is not expected_function
        ):
            raise ValueError
    except BaseException:
        raise ValueError(_UNTRUSTED_TRANSLATOR) from None
    return translator


def build_schema_probes(
    source_adapter: AppWorldSemanticAdapter,
    candidate_adapter: SemanticAdapter,
    canonical_call_to_surface: Callable[[Mapping[str, JSONValue]], Mapping[str, JSONValue]],
) -> tuple[SchemaProbe, ...]:
    """Build one source-oracled, non-executed probe per candidate tool."""

    if type(source_adapter) is not AppWorldSemanticAdapter or not isinstance(
        candidate_adapter, SemanticAdapter
    ):
        raise ValueError(_INVALID_PAIR)
    translator = _require_trusted_translator(
        source_adapter,
        candidate_adapter,
        canonical_call_to_surface,
    )

    try:
        source_calls = build_minimal_source_calls(source_adapter)
        if len(source_calls) != len(source_adapter.variant.tools):
            raise ValueError

        probes: list[SchemaProbe] = []
        surface_call_bytes: list[bytes] = []
        for index, canonical_call in enumerate(source_calls):
            expected_actions = source_adapter.surface_to_semantic(canonical_call)
            first_surface_call = translator(canonical_call)
            second_surface_call = translator(canonical_call)
            first_bytes = canonical_json_bytes(first_surface_call)
            if first_bytes != canonical_json_bytes(second_surface_call):
                raise ValueError
            surface_name = first_surface_call.get("name")
            if type(surface_name) is not str:
                raise ValueError
            probes.append(
                SchemaProbe(
                    f"schema-{index:04d}",
                    surface_name,
                    first_surface_call,
                    expected_actions,
                )
            )
            surface_call_bytes.append(first_bytes)

        candidate_names = {tool.name for tool in candidate_adapter.variant.tools}
        probe_names = {probe.surface_tool_name for probe in probes}
        if (
            len(probes) != len(candidate_adapter.variant.tools)
            or len(probe_names) != len(probes)
            or len(set(surface_call_bytes)) != len(surface_call_bytes)
            or probe_names != candidate_names
        ):
            raise ValueError
        return tuple(probes)
    except BaseException:
        raise ValueError(_SCHEMA_PROBE_FAILURE) from None


def _json_matches(first: JSONValue, second: JSONValue) -> bool:
    return canonical_json_bytes(first) == canonical_json_bytes(second)


def _fail_evidence(code: str) -> None:
    raise _VariantAdmissionError(_PAIR_EVIDENCE_FAILURE, (code,))


def _require_paired_steps_match(reference: object, candidate: object) -> None:
    try:
        reference_action = reference.action
        candidate_action = candidate.action
        actions_match = (
            type(reference_action.name) is str
            and type(candidate_action.name) is str
            and reference_action.name == candidate_action.name
            and canonical_json_bytes(reference_action.arguments)
            == canonical_json_bytes(candidate_action.arguments)
        )
    except BaseException:
        _fail_evidence("pair.action_mismatch")
    if not actions_match:
        _fail_evidence("pair.action_mismatch")
    for field_name, code in (
        ("base_call", "pair.base_call_mismatch"),
        ("base_observation", "pair.base_observation_mismatch"),
        ("surface_observation", "pair.surface_observation_mismatch"),
    ):
        try:
            matches = _json_matches(
                getattr(reference, field_name),
                getattr(candidate, field_name),
            )
        except BaseException:
            _fail_evidence(code)
        if not matches:
            _fail_evidence(code)


def _execute_fresh(
    factory: _FreshWorldFactory,
    *,
    role: str,
    adapter: SemanticAdapter,
    calls: tuple[Mapping[str, JSONValue], ...],
    seen_worlds: list[object],
) -> object:
    with factory(role=role) as world:
        if any(world is seen for seen in seen_worlds):
            raise _VariantAdmissionError(
                _FRESH_WORLD_FAILURE,
                ("pair.world_not_fresh",),
            )
        seen_worlds.append(world)
        return AppWorldEpisodeExecutor(world, adapter).execute_plan(calls)


def _fresh_initial_state(
    factory: _FreshWorldFactory,
    *,
    role: str,
    seen_worlds: list[object],
) -> str:
    with factory(role=role) as world:
        if any(world is seen for seen in seen_worlds):
            raise _VariantAdmissionError(
                _FRESH_WORLD_FAILURE,
                ("pair.world_not_fresh",),
            )
        seen_worlds.append(world)
        try:
            return canonical_state_sha256(world.models)
        except BaseException:
            raise _VariantAdmissionError(
                _PAIR_EXECUTION_FAILURE,
                ("pair.reset_hash_exception",),
            ) from None


def _denotation_cases(
    reference_record: object,
    candidate_record: object,
) -> tuple[DenotationCase, ...]:
    reference_steps = reference_record.steps
    candidate_steps = candidate_record.steps
    if (
        type(reference_steps) is not tuple
        or type(candidate_steps) is not tuple
        or not reference_steps
        or len(reference_steps) != len(candidate_steps)
    ):
        _fail_evidence("pair.step_count_mismatch")

    cases: list[DenotationCase] = []
    for index, (reference, candidate) in enumerate(
        zip(reference_steps, candidate_steps, strict=True)
    ):
        _require_paired_steps_match(reference, candidate)
        cases.append(
            DenotationCase(
                f"denotation-{index:04d}",
                candidate.surface_call,
                (reference.action,),
                ((reference.base_call,),),
                ((reference.base_observation,),),
                reference.surface_observation,
            )
        )
    return tuple(cases)


def _admit_variant_pair(
    *,
    source_adapter: AppWorldSemanticAdapter,
    candidate_adapter: SemanticAdapter,
    oracle_plan: _CapturedOraclePlan,
    canonical_call_to_surface: Callable[[Mapping[str, JSONValue]], Mapping[str, JSONValue]],
    world_context_factory: _FreshWorldFactory,
) -> _VariantAdmissionRecord:
    """Execute, materialize, and immediately admit one private paired suite."""

    if (
        type(source_adapter) is not AppWorldSemanticAdapter
        or not isinstance(candidate_adapter, SemanticAdapter)
        or not callable(canonical_call_to_surface)
        or not callable(world_context_factory)
    ):
        raise ValueError(_INVALID_PAIR) from None
    if type(oracle_plan) is not _CapturedOraclePlan:
        raise ValueError(_INVALID_PLAN) from None
    translator = _require_trusted_translator(
        source_adapter,
        candidate_adapter,
        canonical_call_to_surface,
    )

    try:
        oracle_native_calls = oracle_plan.native_calls
        if type(oracle_native_calls) is not tuple or not oracle_native_calls:
            raise _VariantAdmissionError(_INVALID_PLAN, ("pair.invalid_oracle_plan",))
        schema_probes = build_schema_probes(
            source_adapter,
            candidate_adapter,
            translator,
        )
        first_candidate_calls = tuple(translator(call) for call in oracle_native_calls)
        second_candidate_calls = tuple(translator(call) for call in oracle_native_calls)
        if canonical_json_bytes(first_candidate_calls) != canonical_json_bytes(
            second_candidate_calls
        ):
            _fail_evidence("pair.translation_nondeterministic")
        candidate_calls = first_candidate_calls
        seen_worlds: list[object] = []
        reference_record = _execute_fresh(
            world_context_factory,
            role="reference",
            adapter=source_adapter,
            calls=oracle_native_calls,
            seen_worlds=seen_worlds,
        )
        candidate_record = _execute_fresh(
            world_context_factory,
            role="candidate",
            adapter=candidate_adapter,
            calls=candidate_calls,
            seen_worlds=seen_worlds,
        )
        if reference_record.evaluator_score.get("success") is not True:
            _fail_evidence("pair.reference_oracle_unsuccessful")
        if candidate_record.evaluator_score.get("success") is not True:
            _fail_evidence("pair.candidate_oracle_unsuccessful")
        denotation_cases = _denotation_cases(reference_record, candidate_record)
        reference_reset = _fresh_initial_state(
            world_context_factory,
            role="reference-reset",
            seen_worlds=seen_worlds,
        )
        candidate_reset = _fresh_initial_state(
            world_context_factory,
            role="candidate-reset",
            seen_worlds=seen_worlds,
        )

        state_case = StateCase("state-0000", "episode-0000")
        trace_case = TraceCase("trace-0000", "episode-0000")
        state_evidence = StateEvidence(
            reference_record.initial_state_sha256,
            candidate_record.initial_state_sha256,
            reference_record.final_state_sha256,
            candidate_record.final_state_sha256,
            reference_reset,
            candidate_reset,
            reference_record.transition_digest,
            candidate_record.transition_digest,
        )
        trace_evidence = TraceEvidence(
            reference_record.trace,
            candidate_record.trace,
            denotation_cases,
            reference_record.evaluator_score,
            candidate_record.evaluator_score,
            reference_record.effects,
            candidate_record.effects,
        )
        suite = evaluate_contract_suite(
            variant=candidate_adapter.variant,
            adapter=candidate_adapter,
            schema_probes=schema_probes,
            denotation_cases=denotation_cases,
            state_cases=(state_case,),
            state_provider=_StateEvidenceLookup(((state_case, state_evidence),)),
            trace_cases=(trace_case,),
            trace_provider=_TraceEvidenceLookup(((trace_case, trace_evidence),)),
        )
        checks_run = tuple((result.layer, result.checks_run) for result in suite.results)
        diagnostic_codes = tuple(
            diagnostic.code for result in suite.results for diagnostic in result.diagnostics
        )
        try:
            admitted = require_dataset_admission(candidate_adapter.variant, suite)
        except BaseException:
            codes = diagnostic_codes or ("pair.admission_failed",)
            raise _VariantAdmissionError(_PAIR_EVIDENCE_FAILURE, codes) from None
        if admitted is not suite:
            raise _VariantAdmissionError(
                _PAIR_EVIDENCE_FAILURE,
                ("pair.admission_identity_mismatch",),
            )
        return _VariantAdmissionRecord(
            1,
            checks_run,
            (),
        )
    except _VariantAdmissionError as error:
        raise error from None
    except BaseException:
        raise _VariantAdmissionError(
            _PAIR_EXECUTION_FAILURE,
            ("pair.execution_exception",),
        ) from None


__all__ = ["build_schema_probes"]
