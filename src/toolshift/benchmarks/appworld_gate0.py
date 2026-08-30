"""Process-local paired evidence for the AppWorld Gate 0 smoke.

This module owns no AppWorld lifecycle globally and deliberately persists no
contract suite, episode record, task identity, call, observation, or digest.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Protocol
from urllib.parse import unquote, urlparse

from toolshift.adapters import (
    AppWorldSemanticAdapter,
    build_appworld_adapter,
    build_minimal_source_calls,
)
from toolshift.adapters.semantic import SemanticAdapter
from toolshift.benchmarks.appworld_replay import (
    PINNED_APPWORLD_COMMIT,
    _AppWorldCleanupMarker,
    appworld_world_context,
    canonical_state_sha256,
)
from toolshift.benchmarks.appworld_runtime import (
    AppWorldEpisodeExecutor,
    _capture_oracle_plan,
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
from toolshift.transforms import (
    ParameterGroupRule,
    ParameterRestructureAdapter,
    RenameAdapter,
    TransformValidationError,
    apply_parameter_restructure,
    apply_rename,
)
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
_PREFLIGHT_FAILURE = "AppWorld Gate 0 preflight failed"
_WORLD_FAILURE = "AppWorld Gate 0 world lifecycle failed"
_SAFE_DIAGNOSTIC = "gate0.unsafe_diagnostic"
_PINNED_APPWORLD_PACKAGE_VERSION = "0.2.0.dev0"
_PINNED_APPWORLD_DATA_VERSION = "0.2.0"
_PINNED_APPWORLD_DB_VERSION = "0.2.0"
_PINNED_PYTHON_VERSION = (3, 11, 15)
_SMOKE_ROLES = frozenset({"screen", "reference", "candidate", "reference-reset", "candidate-reset"})
_WORLD_FLAGS = MappingProxyType(
    {
        "raise_on_failure": False,
        "raise_on_extra_parameters": True,
        "remote_apis_url": None,
        "remote_environment_url": None,
        "remote_mcp_url": None,
        "remote_docker": False,
        "parse_datetimes": False,
        "wrap_response": False,
        "unwrap_response": False,
        "munchify_response": False,
    }
)
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


def _require_exact_nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("AppWorld Gate 0 summary is invalid")
    return value


def _is_safe_diagnostic_code(value: object) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= 96
        and all(
            character.isascii() and (character.isalnum() or character in "._-")
            for character in value
        )
    )


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class AppWorldTransformSummary:
    """Aggregate-only counters for one smoke transform family."""

    requested_count: int
    eligible_count: int
    attempted_count: int
    admitted_count: int
    excluded_count: int

    def __post_init__(self) -> None:
        values = tuple(
            _require_exact_nonnegative_int(getattr(self, name))
            for name in (
                "requested_count",
                "eligible_count",
                "attempted_count",
                "admitted_count",
                "excluded_count",
            )
        )
        requested, eligible, attempted, admitted, _ = values
        if admitted > attempted or attempted > eligible or eligible > requested:
            raise ValueError("AppWorld Gate 0 summary is invalid")


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class AppWorldGate0Summary:
    """Frozen aggregate status; protected task evidence never enters this model."""

    mode: str
    split: str
    seed: int
    workers: int
    screened_task_count: int
    excluded_task_count: int
    admitted_task_count: int
    clean: AppWorldTransformSummary
    l1: AppWorldTransformSummary
    l2: AppWorldTransformSummary
    diagnostic_counts: tuple[tuple[str, int], ...]
    execution_exception_count: int
    cleanup_exception_count: int
    gate_evaluable: bool
    formal_gate_passed: bool
    smoke_passed: bool

    def __post_init__(self) -> None:
        if (
            self.mode != "smoke"
            or self.split != "train"
            or type(self.seed) is not int
            or type(self.workers) is not int
            or self.workers != 1
            or type(self.clean) is not AppWorldTransformSummary
            or type(self.l1) is not AppWorldTransformSummary
            or type(self.l2) is not AppWorldTransformSummary
            or type(self.gate_evaluable) is not bool
            or self.gate_evaluable
            or type(self.formal_gate_passed) is not bool
            or self.formal_gate_passed
            or type(self.smoke_passed) is not bool
        ):
            raise ValueError("AppWorld Gate 0 summary is invalid")
        for value in (
            self.screened_task_count,
            self.excluded_task_count,
            self.admitted_task_count,
            self.execution_exception_count,
            self.cleanup_exception_count,
        ):
            _require_exact_nonnegative_int(value)
        if type(self.diagnostic_counts) is not tuple:
            raise ValueError("AppWorld Gate 0 summary is invalid")
        previous = ""
        for entry in self.diagnostic_counts:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("AppWorld Gate 0 summary is invalid")
            code, count = entry
            if not _is_safe_diagnostic_code(code) or code <= previous:
                raise ValueError("AppWorld Gate 0 summary is invalid")
            if type(count) is not int or count <= 0:
                raise ValueError("AppWorld Gate 0 summary is invalid")
            previous = code
        expected_pass = (
            self.admitted_task_count == 1
            and self.clean.admitted_count == 1
            and self.l1.admitted_count == 1
            and self.l2.admitted_count == 1
            and not self.diagnostic_counts
            and self.execution_exception_count == 0
            and self.cleanup_exception_count == 0
        )
        if self.smoke_passed is not expected_pass:
            raise ValueError("AppWorld Gate 0 summary is invalid")


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


class _SmokeAccounting:
    __slots__ = (
        "admitted_task_count",
        "cleanup_exception_count",
        "diagnostics",
        "execution_exception_count",
        "families",
        "screened_task_count",
    )

    def __init__(self) -> None:
        self.screened_task_count = 0
        self.admitted_task_count = 0
        self.execution_exception_count = 0
        self.cleanup_exception_count = 0
        self.diagnostics: Counter[str] = Counter()
        self.families = {
            family: {
                "requested": 0,
                "eligible": 0,
                "attempted": 0,
                "admitted": 0,
            }
            for family in ("clean", "l1", "l2")
        }

    def add_diagnostic(self, code: object) -> None:
        safe_code = code if _is_safe_diagnostic_code(code) else _SAFE_DIAGNOSTIC
        self.diagnostics[safe_code] += 1

    def summary(self, *, seed: int, workers: int) -> AppWorldGate0Summary:
        family_summaries: dict[str, AppWorldTransformSummary] = {}
        for family in ("clean", "l1", "l2"):
            counters = self.families[family]
            requested = counters["requested"]
            eligible = counters["eligible"]
            family_summaries[family] = AppWorldTransformSummary(
                requested,
                eligible,
                counters["attempted"],
                counters["admitted"],
                requested - eligible,
            )
        diagnostic_counts = tuple(sorted(self.diagnostics.items()))
        smoke_passed = (
            self.admitted_task_count == 1
            and all(
                family_summaries[family].admitted_count == 1 for family in ("clean", "l1", "l2")
            )
            and not diagnostic_counts
            and self.execution_exception_count == 0
            and self.cleanup_exception_count == 0
        )
        return AppWorldGate0Summary(
            mode="smoke",
            split="train",
            seed=seed,
            workers=workers,
            screened_task_count=self.screened_task_count,
            excluded_task_count=(self.screened_task_count - self.admitted_task_count),
            admitted_task_count=self.admitted_task_count,
            clean=family_summaries["clean"],
            l1=family_summaries["l1"],
            l2=family_summaries["l2"],
            diagnostic_counts=diagnostic_counts,
            execution_exception_count=self.execution_exception_count,
            cleanup_exception_count=self.cleanup_exception_count,
            gate_evaluable=False,
            formal_gate_passed=False,
            smoke_passed=smoke_passed,
        )


def _has_cleanup_marker(error: BaseException) -> bool:
    marker: BaseException | None = error
    for _ in range(3):
        if type(marker) is _AppWorldCleanupMarker:
            return True
        marker = marker.__cause__
        if marker is None:
            return False
    return False


class _SmokeWorldContexts:
    __slots__ = (
        "_accounting",
        "_context_count",
        "_factory",
        "_live",
        "_seed",
        "_seen_worlds",
    )

    def __init__(
        self,
        factory: Callable[..., AbstractContextManager[object]],
        *,
        seed: int,
        accounting: _SmokeAccounting,
    ) -> None:
        self._factory = factory
        self._seed = seed
        self._accounting = accounting
        self._context_count = 0
        self._live = False
        self._seen_worlds: list[object] = []

    @contextmanager
    def open(self, *, task_id: str, role: str) -> Iterator[object]:
        if type(task_id) is not str or not task_id or role not in _SMOKE_ROLES:
            raise ValueError(_WORLD_FAILURE)
        sequence = self._context_count
        self._context_count += 1
        kwargs: dict[str, object] = {
            "task_id": task_id,
            "experiment_name": f"toolshift-m3a-{sequence:06d}-{uuid.uuid4().hex}",
            "random_seed": self._seed,
            **_WORLD_FLAGS,
        }
        if role in {"screen", "reference"}:
            kwargs["ground_truth_mode"] = "full"

        entered = False
        body_failed = False
        body_completed = False
        try:
            with self._factory(**kwargs) as world:
                entered = True
                if self._live or any(world is previous for previous in self._seen_worlds):
                    raise ValueError(_WORLD_FAILURE)
                self._seen_worlds.append(world)
                self._live = True
                try:
                    yield world
                except BaseException:
                    body_failed = True
                    raise
                else:
                    body_completed = True
                finally:
                    self._live = False
        except BaseException as error:
            cleanup_failed = _has_cleanup_marker(error) or (
                entered and body_completed and not body_failed
            )
            if cleanup_failed:
                self._accounting.cleanup_exception_count += 1
            if body_failed or not cleanup_failed or not entered:
                self._accounting.execution_exception_count += 1
            raise


class _BoundPairWorldFactory:
    __slots__ = ("_contexts", "_task_id")

    def __init__(self, contexts: _SmokeWorldContexts, task_id: str) -> None:
        self._contexts = contexts
        self._task_id = task_id

    def __call__(self, *, role: str) -> AbstractContextManager[object]:
        return self._contexts.open(task_id=self._task_id, role=role)


def _load_appworld_task_ids(split: str) -> Sequence[str]:
    if split != "train":
        raise RuntimeError(_PREFLIGHT_FAILURE)
    try:
        from appworld import load_task_ids

        return load_task_ids("train")
    except BaseException:
        raise RuntimeError(_PREFLIGHT_FAILURE) from None


def _require_private_appworld_root() -> Path:
    try:
        raw_root = os.environ["APPWORLD_ROOT"]
        root = Path(raw_root)
        root_stat = root.lstat()
        resolved = root.resolve(strict=True)
        if (
            type(raw_root) is not str
            or not raw_root
            or not root.is_absolute()
            or root != resolved
            or stat.S_ISLNK(root_stat.st_mode)
            or not stat.S_ISDIR(root_stat.st_mode)
            or stat.S_IMODE(root_stat.st_mode) != 0o700
            or root_stat.st_uid != os.geteuid()
            or any((parent / ".git").exists() for parent in (root, *root.parents))
        ):
            raise ValueError
        return root
    except BaseException:
        raise RuntimeError(_PREFLIGHT_FAILURE) from None


def _require_editable_appworld_checkout() -> Path:
    try:
        distribution = importlib.metadata.distribution("appworld")
        if distribution.version != _PINNED_APPWORLD_PACKAGE_VERSION:
            raise ValueError
        direct_url_text = distribution.read_text("direct_url.json")
        if type(direct_url_text) is not str:
            raise ValueError
        direct_url = json.loads(direct_url_text)
        if type(direct_url) is not dict:
            raise ValueError
        directory_info = direct_url.get("dir_info")
        url = direct_url.get("url")
        if (
            type(directory_info) is not dict
            or directory_info.get("editable") is not True
            or type(url) is not str
        ):
            raise ValueError
        parsed = urlparse(url)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            raise ValueError
        checkout = Path(unquote(parsed.path)).resolve(strict=True)
        if not checkout.is_absolute() or not (checkout / ".git").exists():
            raise ValueError
        completed = subprocess.run(
            ["git", "-C", os.fspath(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.stdout.strip() != PINNED_APPWORLD_COMMIT:
            raise ValueError
        symbolic = subprocess.run(
            ["git", "-C", os.fspath(checkout), "symbolic-ref", "-q", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if symbolic.returncode != 1:
            raise ValueError
        return checkout
    except BaseException:
        raise RuntimeError(_PREFLIGHT_FAILURE) from None


def require_pinned_appworld_runtime() -> None:
    """Validate the private pinned runtime without exposing paths or values."""

    try:
        if any(key.lower().endswith("_proxy") for key in os.environ):
            raise ValueError
        if tuple(sys.version_info[:3]) != _PINNED_PYTHON_VERSION:
            raise ValueError
        root = _require_private_appworld_root()
        checkout = _require_editable_appworld_checkout()
        package_module = importlib.import_module("appworld")
        constants_module = importlib.import_module("appworld.common.constants")
        package_root = (checkout / "src" / "appworld").resolve(strict=True)
        for module in (package_module, constants_module):
            module_file = getattr(module, "__file__", None)
            if type(module_file) is not str:
                raise ValueError
            Path(module_file).resolve(strict=True).relative_to(package_root)

        if (
            getattr(constants_module, "DATA_VERSION", None) != _PINNED_APPWORLD_DATA_VERSION
            or getattr(constants_module, "DB_VERSION", None) != _PINNED_APPWORLD_DB_VERSION
            or (root / "data" / "version.txt").read_text(encoding="utf-8").strip()
            != _PINNED_APPWORLD_DATA_VERSION
            or (root / "data" / "base_dbs" / "version.txt").read_text(encoding="utf-8").strip()
            != _PINNED_APPWORLD_DB_VERSION
        ):
            raise ValueError
    except BaseException:
        raise RuntimeError(_PREFLIGHT_FAILURE) from None


def _ordered_oracle_tool_names(
    oracle_plan: _CapturedOraclePlan,
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for call in oracle_plan.native_calls:
        name = call.get("name")
        if type(name) is not str:
            raise ValueError(_WORLD_FAILURE)
        if name not in seen:
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError(_WORLD_FAILURE)
    return tuple(names)


def _build_l1_candidate(
    source_adapter: AppWorldSemanticAdapter,
    oracle_tool_names: tuple[str, ...],
    *,
    seed: int,
) -> RenameAdapter:
    existing_names = {tool.name for tool in source_adapter.variant.tools}
    alias_index = 0
    while True:
        alias = f"toolshift_l1_{alias_index:04d}"
        if alias not in existing_names:
            break
        alias_index += 1
    return apply_rename(
        source_adapter,
        tool_name_mapping={oracle_tool_names[0]: alias},
        seed=seed,
    )


def _build_l2_candidate(
    source_adapter: AppWorldSemanticAdapter,
    oracle_tool_names: tuple[str, ...],
    *,
    seed: int,
) -> ParameterRestructureAdapter | None:
    tools = {tool.name: tool for tool in source_adapter.variant.tools}
    for tool_name in oracle_tool_names:
        tool = tools.get(tool_name)
        if tool is None:
            raise ValueError(_WORLD_FAILURE)
        properties = tool.input_schema.get("properties")
        if not isinstance(properties, Mapping):
            raise ValueError(_WORLD_FAILURE)
        property_names = tuple(sorted(properties))
        if len(property_names) < 2:
            continue
        for parameter_name in property_names:
            container_index = 0
            while True:
                container_name = f"toolshift_group_{container_index:04d}"
                if container_name not in properties:
                    break
                container_index += 1
            try:
                return apply_parameter_restructure(
                    source_adapter,
                    rules=(
                        ParameterGroupRule(
                            tool_name,
                            container_name,
                            (parameter_name,),
                        ),
                    ),
                    seed=seed,
                )
            except TransformValidationError:
                continue
    return None


def _require_task_ids(value: object) -> tuple[str, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(_PREFLIGHT_FAILURE)
    task_ids = tuple(value)
    if any(type(task_id) is not str or not task_id for task_id in task_ids):
        raise ValueError(_PREFLIGHT_FAILURE)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(_PREFLIGHT_FAILURE)
    return task_ids


def run_appworld_gate0_smoke(
    *,
    seed: int = 100,
    workers: int = 1,
    task_loader: Callable[[str], Sequence[str]] = _load_appworld_task_ids,
    world_context_factory: Callable[..., AbstractContextManager[object]] = (appworld_world_context),
    pin_checker: Callable[[], None] = require_pinned_appworld_runtime,
) -> AppWorldGate0Summary:
    """Run one private train-only clean/L1/L2 integration smoke."""

    if (
        type(seed) is not int
        or seed < 0
        or type(workers) is not int
        or workers != 1
        or not callable(task_loader)
        or not callable(world_context_factory)
        or not callable(pin_checker)
    ):
        raise ValueError(_PREFLIGHT_FAILURE) from None
    try:
        pin_checker()
    except BaseException:
        raise RuntimeError(_PREFLIGHT_FAILURE) from None

    accounting = _SmokeAccounting()
    contexts = _SmokeWorldContexts(
        world_context_factory,
        seed=seed,
        accounting=accounting,
    )
    try:
        task_ids = _require_task_ids(task_loader("train"))
    except BaseException:
        accounting.add_diagnostic("smoke.task_loader_failure")
        return accounting.summary(seed=seed, workers=workers)

    selected: (
        tuple[
            str,
            AppWorldSemanticAdapter,
            _CapturedOraclePlan,
            RenameAdapter,
            ParameterRestructureAdapter,
        ]
        | None
    ) = None
    for task_id in task_ids:
        accounting.screened_task_count += 1
        for family in ("clean", "l1", "l2"):
            accounting.families[family]["requested"] += 1

        screen_status = "ok"
        source_adapter: AppWorldSemanticAdapter | None = None
        oracle_plan: _CapturedOraclePlan | None = None
        try:
            with contexts.open(task_id=task_id, role="screen") as world:
                try:
                    catalog = world.task.api_docs.function_calling()
                    source_adapter = build_appworld_adapter(catalog)
                except BaseException:
                    screen_status = "catalog_invalid"
                if source_adapter is not None:
                    try:
                        build_minimal_source_calls(source_adapter)
                    except BaseException:
                        screen_status = "catalog_unprobeable"
                if screen_status == "ok":
                    try:
                        oracle_plan = _capture_oracle_plan(world)
                    except BaseException:
                        screen_status = "oracle_capture_failed"
        except BaseException:
            accounting.add_diagnostic("smoke.world_lifecycle_failure")
            return accounting.summary(seed=seed, workers=workers)

        if screen_status == "catalog_invalid":
            accounting.add_diagnostic("screen.catalog_invalid")
            continue
        if screen_status == "catalog_unprobeable":
            accounting.add_diagnostic("screen.catalog_unprobeable")
            continue
        if screen_status != "ok" or source_adapter is None or oracle_plan is None:
            accounting.add_diagnostic("screen.oracle_capture_failed")
            return accounting.summary(seed=seed, workers=workers)

        try:
            oracle_tool_names = _ordered_oracle_tool_names(oracle_plan)
            l1_candidate = _build_l1_candidate(
                source_adapter,
                oracle_tool_names,
                seed=seed,
            )
            l2_candidate = _build_l2_candidate(
                source_adapter,
                oracle_tool_names,
                seed=seed,
            )
        except BaseException:
            accounting.add_diagnostic("screen.variant_construction_failed")
            return accounting.summary(seed=seed, workers=workers)
        accounting.families["clean"]["eligible"] += 1
        accounting.families["l1"]["eligible"] += 1
        if l2_candidate is None:
            continue
        accounting.families["l2"]["eligible"] += 1
        selected = (
            task_id,
            source_adapter,
            oracle_plan,
            l1_candidate,
            l2_candidate,
        )
        break

    if selected is None:
        accounting.add_diagnostic("smoke.no_l2_eligible_task")
        return accounting.summary(seed=seed, workers=workers)

    task_id, source_adapter, oracle_plan, l1_candidate, l2_candidate = selected
    pair_factory = _BoundPairWorldFactory(contexts, task_id)
    variants: tuple[
        tuple[
            str,
            SemanticAdapter,
            Callable[[Mapping[str, JSONValue]], Mapping[str, JSONValue]],
        ],
        ...,
    ] = (
        ("clean", source_adapter, _identity_canonical_call_to_surface),
        ("l1", l1_candidate, l1_candidate.transform.canonical_call_to_surface),
        ("l2", l2_candidate, l2_candidate.transform.canonical_call_to_surface),
    )
    for family, candidate_adapter, translator in variants:
        accounting.families[family]["attempted"] += 1
        try:
            record = _admit_variant_pair(
                source_adapter=source_adapter,
                candidate_adapter=candidate_adapter,
                oracle_plan=oracle_plan,
                canonical_call_to_surface=translator,
                world_context_factory=pair_factory,
            )
        except _VariantAdmissionError as error:
            for code in error.diagnostic_codes:
                accounting.add_diagnostic(code)
            return accounting.summary(seed=seed, workers=workers)
        except BaseException:
            accounting.add_diagnostic("smoke.admission_exception")
            return accounting.summary(seed=seed, workers=workers)
        if (
            type(record) is not _VariantAdmissionRecord
            or record.admitted_suite_count != 1
            or record.diagnostic_codes
        ):
            accounting.add_diagnostic("smoke.invalid_admission_record")
            return accounting.summary(seed=seed, workers=workers)
        accounting.families[family]["admitted"] += 1

    accounting.admitted_task_count = 1
    return accounting.summary(seed=seed, workers=workers)


__all__ = [
    "AppWorldGate0Summary",
    "AppWorldTransformSummary",
    "build_schema_probes",
    "require_pinned_appworld_runtime",
    "run_appworld_gate0_smoke",
]
