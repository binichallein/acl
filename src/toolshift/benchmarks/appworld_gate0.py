"""Process-local paired evidence for the AppWorld Gate 0 smoke.

This module owns no AppWorld lifecycle globally and deliberately persists no
contract suite, episode record, task identity, call, observation, or digest.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import stat
import subprocess
import sys
import threading
import uuid
import zipfile
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
from toolshift.benchmarks._evidence import write_private_json_beneath
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
_PINNED_PYTHON_DOTENV_VERSION = "1.2.2"
_PINNED_APPWORLD_BUNDLE_LAYOUT = (
    (b"src/appworld/.source/apps.bundle", b"src/appworld"),
    (b"src/appworld/.source/tests.bundle", b"tests"),
    (b"generate/.source/tasks.bundle", b"generate/tasks"),
    (b"generate/.source/data.bundle", b"generate"),
)
_FORBIDDEN_APPWORLD_RUNTIME_ENVIRONMENT = frozenset(
    {"APPWORLD_DB_ARGS", "APPWORLD_DATE_TIME", "LOAD_ON_STARTUP"}
)
APPWORLD_ADAPTER_ABI_SHA256 = hashlib.sha256(b"toolshift.appworld.adapter.v1").hexdigest()
APPWORLD_GATE0_RUNNER_ABI_SHA256 = hashlib.sha256(b"toolshift.appworld.gate0.runner.v1").hexdigest()
_SMOKE_ROLES = frozenset({"screen", "reference", "candidate", "reference-reset", "candidate-reset"})
_KNOWN_GATE0_DIAGNOSTIC_CODES = frozenset(
    {
        "denotation.base_group_count",
        "denotation.compile_exception",
        "denotation.compile_mismatch",
        "denotation.duplicate_case",
        "denotation.empty_cases",
        "denotation.invalid_adapter",
        "denotation.invalid_case",
        "denotation.invalid_cases",
        "denotation.nondeterministic_compile",
        "denotation.nondeterministic_observation",
        "denotation.nondeterministic_parse",
        "denotation.observation_cardinality",
        "denotation.observation_exception",
        "denotation.observation_group_count",
        "denotation.observation_mismatch",
        "denotation.parse_exception",
        "denotation.parse_mismatch",
        "denotation.variant_rebound",
        "gate0.unsafe_diagnostic",
        "pair.action_mismatch",
        "pair.admission_failed",
        "pair.admission_identity_mismatch",
        "pair.base_call_mismatch",
        "pair.base_observation_mismatch",
        "pair.candidate_oracle_unsuccessful",
        "pair.execution_exception",
        "pair.invalid_oracle_plan",
        "pair.reference_oracle_unsuccessful",
        "pair.reset_hash_exception",
        "pair.step_count_mismatch",
        "pair.surface_observation_mismatch",
        "pair.translation_nondeterministic",
        "pair.unsafe_diagnostic",
        "pair.world_not_fresh",
        "schema.call_name_mismatch",
        "schema.duplicate_call",
        "schema.duplicate_case",
        "schema.empty_probes",
        "schema.invalid_probe",
        "schema.invalid_probes",
        "schema.invalid_variant",
        "schema.mapping_exception",
        "schema.mapping_mismatch",
        "schema.missing_tool",
        "schema.nondeterministic_mapping",
        "schema.undeclared_tool",
        "schema.variant_exception",
        "schema.variant_mismatch",
        "schema.variant_rebound",
        "screen.catalog_invalid",
        "screen.catalog_unprobeable",
        "screen.oracle_capture_failed",
        "screen.variant_construction_failed",
        "smoke.admission_exception",
        "smoke.invalid_admission_record",
        "smoke.no_l2_eligible_task",
        "smoke.preflight_failure",
        "smoke.task_loader_failure",
        "smoke.world_lifecycle_failure",
        "state.candidate_reset_mismatch",
        "state.collateral_mismatch",
        "state.duplicate_case",
        "state.duplicate_episode",
        "state.empty_cases",
        "state.final_mismatch",
        "state.initial_mismatch",
        "state.invalid_case",
        "state.invalid_cases",
        "state.provider_exception",
        "state.reference_reset_mismatch",
        "suite.case_snapshot_mismatch",
        "suite.denotation_exception",
        "suite.episode_set_mismatch",
        "suite.schema_exception",
        "suite.state_exception",
        "suite.trace_exception",
        "suite.variant_binding_mismatch",
        "trace.base_call_mismatch",
        "trace.base_reconstruction_mismatch",
        "trace.canonical_semantic_channel_mismatch",
        "trace.canonicalization_exception",
        "trace.duplicate_case",
        "trace.duplicate_episode",
        "trace.duplicate_verified_step",
        "trace.effect_mismatch",
        "trace.empty_cases",
        "trace.empty_steps",
        "trace.candidate_effect_alignment",
        "trace.candidate_effect_count",
        "trace.invalid_adapter",
        "trace.invalid_case",
        "trace.invalid_cases",
        "trace.invalid_step_structure",
        "trace.invalid_verified_step",
        "trace.invalid_verified_steps",
        "trace.provider_exception",
        "trace.reference_effect_alignment",
        "trace.reference_effect_count",
        "trace.score_mismatch",
        "trace.semantic_mismatch",
        "trace.semantic_reconstruction_mismatch",
        "trace.surface_reconstruction_mismatch",
        "trace.unverified_step",
        "trace.variant_rebound",
        "trace.verified_step_mutated",
    }
)
_ZERO_SCREEN_DIAGNOSTIC_CODES = frozenset({"smoke.preflight_failure", "smoke.task_loader_failure"})
_SCREEN_EXCLUSION_DIAGNOSTIC_CODES = frozenset(
    {"screen.catalog_invalid", "screen.catalog_unprobeable"}
)
_SCREEN_TERMINAL_DIAGNOSTIC_CODES = frozenset(
    {
        "screen.oracle_capture_failed",
        "screen.variant_construction_failed",
        "smoke.world_lifecycle_failure",
    }
)
_NO_L2_DIAGNOSTIC_CODE = "smoke.no_l2_eligible_task"
_ADMISSION_DIAGNOSTIC_CODES = _KNOWN_GATE0_DIAGNOSTIC_CODES.difference(
    _ZERO_SCREEN_DIAGNOSTIC_CODES,
    _SCREEN_EXCLUSION_DIAGNOSTIC_CODES,
    _SCREEN_TERMINAL_DIAGNOSTIC_CODES,
    {_NO_L2_DIAGNOSTIC_CODE},
)
_SINGLETON_DIAGNOSTIC_CODES = frozenset(
    {
        *_ZERO_SCREEN_DIAGNOSTIC_CODES,
        *_SCREEN_TERMINAL_DIAGNOSTIC_CODES,
        _NO_L2_DIAGNOSTIC_CODE,
        "gate0.unsafe_diagnostic",
        "smoke.admission_exception",
        "smoke.invalid_admission_record",
        "schema.duplicate_call",
        "schema.duplicate_case",
        "schema.empty_probes",
        "schema.invalid_probes",
        "schema.invalid_variant",
        "schema.missing_tool",
        "schema.undeclared_tool",
        "schema.variant_exception",
        "schema.variant_mismatch",
        "denotation.duplicate_case",
        "denotation.empty_cases",
        "denotation.invalid_adapter",
        "denotation.invalid_cases",
        "trace.duplicate_case",
        "trace.duplicate_episode",
        "trace.empty_cases",
        "trace.invalid_adapter",
        "trace.invalid_cases",
        "trace.candidate_effect_count",
        "trace.reference_effect_count",
        "suite.denotation_exception",
        "suite.episode_set_mismatch",
        "suite.schema_exception",
        "suite.state_exception",
        "suite.trace_exception",
        *(
            code
            for code in _KNOWN_GATE0_DIAGNOSTIC_CODES
            if code.startswith("pair.") or code.startswith("state.")
        ),
    }
)
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
_SMOKE_WORLD_SLOT = threading.Lock()
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
            sanitized: list[str] = []
            singleton_seen: set[str] = set()
            for code in diagnostic_codes:
                safe_code = code if _is_safe_diagnostic_code(code) else "pair.unsafe_diagnostic"
                if safe_code in _SINGLETON_DIAGNOSTIC_CODES:
                    if safe_code in singleton_seen:
                        continue
                    singleton_seen.add(safe_code)
                sanitized.append(safe_code)
            self.diagnostic_codes = tuple(sanitized)
        except BaseException:
            self.diagnostic_codes = ("pair.unsafe_diagnostic",)

    def __repr__(self) -> str:
        return "_VariantAdmissionError(<private>)"


def _require_exact_nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("AppWorld Gate 0 summary is invalid")
    return value


def _is_safe_diagnostic_code(value: object) -> bool:
    return type(value) is str and value in _KNOWN_GATE0_DIAGNOSTIC_CODES


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class AppWorldTransformSummary:
    """Aggregate-only counters for one smoke transform family."""

    requested_count: int
    eligible_count: int
    attempted_count: int
    admitted_count: int
    excluded_count: int

    def __post_init__(self) -> None:
        self._validate_integrity()

    def _validate_integrity(self) -> None:
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
        if (
            admitted > attempted
            or attempted > eligible
            or eligible > requested
            or self.excluded_count != requested - eligible
        ):
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
    schema_version: int = 1
    pinned_appworld_commit: str = PINNED_APPWORLD_COMMIT
    pinned_appworld_package_version: str = _PINNED_APPWORLD_PACKAGE_VERSION
    pinned_appworld_data_version: str = _PINNED_APPWORLD_DATA_VERSION
    pinned_appworld_base_db_version: str = _PINNED_APPWORLD_DB_VERSION
    pinned_python_version: str = "3.11.15"
    adapter_abi_sha256: str = APPWORLD_ADAPTER_ABI_SHA256
    runner_abi_sha256: str = APPWORLD_GATE0_RUNNER_ABI_SHA256

    def __post_init__(self) -> None:
        self._validate_integrity()

    def _validate_integrity(self) -> None:
        if (
            type(self.mode) is not str
            or self.mode != "smoke"
            or type(self.split) is not str
            or self.split != "train"
            or type(self.seed) is not int
            or self.seed < 0
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
            or type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.pinned_appworld_commit) is not str
            or self.pinned_appworld_commit != PINNED_APPWORLD_COMMIT
            or type(self.pinned_appworld_package_version) is not str
            or self.pinned_appworld_package_version != _PINNED_APPWORLD_PACKAGE_VERSION
            or type(self.pinned_appworld_data_version) is not str
            or self.pinned_appworld_data_version != _PINNED_APPWORLD_DATA_VERSION
            or type(self.pinned_appworld_base_db_version) is not str
            or self.pinned_appworld_base_db_version != _PINNED_APPWORLD_DB_VERSION
            or type(self.pinned_python_version) is not str
            or self.pinned_python_version != "3.11.15"
            or type(self.adapter_abi_sha256) is not str
            or self.adapter_abi_sha256 != APPWORLD_ADAPTER_ABI_SHA256
            or type(self.runner_abi_sha256) is not str
            or self.runner_abi_sha256 != APPWORLD_GATE0_RUNNER_ABI_SHA256
        ):
            raise ValueError("AppWorld Gate 0 summary is invalid")
        self.clean._validate_integrity()
        self.l1._validate_integrity()
        self.l2._validate_integrity()
        for value in (
            self.screened_task_count,
            self.excluded_task_count,
            self.admitted_task_count,
            self.execution_exception_count,
            self.cleanup_exception_count,
        ):
            _require_exact_nonnegative_int(value)
        if (
            self.admitted_task_count > 1
            or self.admitted_task_count > self.screened_task_count
            or self.excluded_task_count != self.screened_task_count - self.admitted_task_count
            or any(
                summary.requested_count != self.screened_task_count
                for summary in (self.clean, self.l1, self.l2)
            )
            or self.clean.eligible_count != self.l1.eligible_count
            or self.l2.eligible_count > 1
            or self.clean.attempted_count != self.l2.eligible_count
            or self.l1.attempted_count != self.clean.admitted_count
            or self.l2.attempted_count != self.l1.admitted_count
            or self.admitted_task_count != self.l2.admitted_count
        ):
            raise ValueError("AppWorld Gate 0 summary is invalid")
        if type(self.diagnostic_counts) is not tuple:
            raise ValueError("AppWorld Gate 0 summary is invalid")
        previous = ""
        diagnostic_map: dict[str, int] = {}
        for entry in self.diagnostic_counts:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("AppWorld Gate 0 summary is invalid")
            code, count = entry
            if not _is_safe_diagnostic_code(code) or code <= previous:
                raise ValueError("AppWorld Gate 0 summary is invalid")
            if type(count) is not int or count <= 0:
                raise ValueError("AppWorld Gate 0 summary is invalid")
            previous = code
            diagnostic_map[code] = count
        if any(
            count != 1
            for code, count in diagnostic_map.items()
            if code in _SINGLETON_DIAGNOSTIC_CODES
        ):
            raise ValueError("AppWorld Gate 0 summary is invalid")
        if any(
            diagnostic_map.get(code, 0) > 4
            for code in ("suite.case_snapshot_mismatch", "suite.variant_binding_mismatch")
        ):
            raise ValueError("AppWorld Gate 0 summary is invalid")
        exception_total = self.execution_exception_count + self.cleanup_exception_count
        if (
            self.execution_exception_count > 1
            or self.cleanup_exception_count > 1
            or (exception_total > 0 and self.admitted_task_count != 0)
        ):
            raise ValueError("AppWorld Gate 0 summary is invalid")
        zero_screen_codes = set(diagnostic_map).intersection(_ZERO_SCREEN_DIAGNOSTIC_CODES)
        exclusion_count = sum(
            diagnostic_map.get(code, 0) for code in _SCREEN_EXCLUSION_DIAGNOSTIC_CODES
        )
        terminal_codes = set(diagnostic_map).intersection(_SCREEN_TERMINAL_DIAGNOSTIC_CODES)
        admission_codes = set(diagnostic_map).intersection(_ADMISSION_DIAGNOSTIC_CODES)
        no_l2 = _NO_L2_DIAGNOSTIC_CODE in diagnostic_map
        if zero_screen_codes:
            if (
                len(zero_screen_codes) != 1
                or len(diagnostic_map) != 1
                or self.screened_task_count != 0
                or self.admitted_task_count != 0
                or exception_total != 0
                or exclusion_count != 0
            ):
                raise ValueError("AppWorld Gate 0 summary is invalid")
        elif no_l2:
            if (
                terminal_codes
                or admission_codes
                or self.clean.attempted_count != 0
                or self.admitted_task_count != 0
                or exception_total != 0
                or exclusion_count != self.screened_task_count - self.clean.eligible_count
            ):
                raise ValueError("AppWorld Gate 0 summary is invalid")
        elif terminal_codes:
            if (
                len(terminal_codes) != 1
                or admission_codes
                or self.clean.attempted_count != 0
                or self.admitted_task_count != 0
                or exclusion_count != self.screened_task_count - self.clean.eligible_count - 1
            ):
                raise ValueError("AppWorld Gate 0 summary is invalid")
            lifecycle_failure = "smoke.world_lifecycle_failure" in terminal_codes
            if (lifecycle_failure and exception_total == 0) or (
                not lifecycle_failure and exception_total != 0
            ):
                raise ValueError("AppWorld Gate 0 summary is invalid")
        elif admission_codes:
            if (
                self.clean.attempted_count != 1
                or self.admitted_task_count != 0
                or exclusion_count != self.screened_task_count - self.clean.eligible_count
            ):
                raise ValueError("AppWorld Gate 0 summary is invalid")
            if exception_total:
                valid_lifecycle_provenance = (
                    admission_codes == {"pair.execution_exception"}
                    or (
                        admission_codes
                        in (
                            {"pair.reset_hash_exception"},
                            {"pair.world_not_fresh"},
                        )
                        and self.execution_exception_count == 1
                    )
                    or (
                        admission_codes == {"smoke.admission_exception"}
                        and self.execution_exception_count == 1
                        and self.cleanup_exception_count == 0
                    )
                )
                if not valid_lifecycle_provenance:
                    raise ValueError("AppWorld Gate 0 summary is invalid")
            elif admission_codes.intersection(
                {"pair.reset_hash_exception", "pair.world_not_fresh"}
            ):
                raise ValueError("AppWorld Gate 0 summary is invalid")
        elif (
            self.admitted_task_count != 1
            or exception_total != 0
            or exclusion_count != self.screened_task_count - self.clean.eligible_count
        ):
            raise ValueError("AppWorld Gate 0 summary is invalid")
        if self.admitted_task_count == 0 and not self.diagnostic_counts:
            raise ValueError("AppWorld Gate 0 summary is invalid")
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

    def to_dict(self) -> dict[str, object]:
        """Return the exact private aggregate allowlist without protected evidence."""

        self._validate_integrity()

        def transform_counts(summary: AppWorldTransformSummary) -> dict[str, int]:
            return {
                "requested": summary.requested_count,
                "eligible": summary.eligible_count,
                "attempted": summary.attempted_count,
                "admitted": summary.admitted_count,
                "excluded": summary.excluded_count,
            }

        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "pinned_appworld_commit": self.pinned_appworld_commit,
            "pinned_appworld_package_version": self.pinned_appworld_package_version,
            "pinned_appworld_data_version": self.pinned_appworld_data_version,
            "pinned_appworld_base_db_version": self.pinned_appworld_base_db_version,
            "pinned_python_version": self.pinned_python_version,
            "adapter_abi_sha256": self.adapter_abi_sha256,
            "runner_abi_sha256": self.runner_abi_sha256,
            "split": self.split,
            "seed": self.seed,
            "workers": self.workers,
            "screened_task_count": self.screened_task_count,
            "admitted_task_count": self.admitted_task_count,
            "transform_counts": {
                "clean": transform_counts(self.clean),
                "l1": transform_counts(self.l1),
                "l2": transform_counts(self.l2),
            },
            "diagnostic_counts": dict(self.diagnostic_counts),
            "execution_exception_count": self.execution_exception_count,
            "cleanup_exception_count": self.cleanup_exception_count,
            "gate_evaluable": self.gate_evaluable,
            "formal_gate_passed": self.formal_gate_passed,
            "smoke_passed": self.smoke_passed,
        }


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
        if safe_code == _SAFE_DIAGNOSTIC:
            self.diagnostics[safe_code] = 1
        else:
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


def _appworld_gate0_preflight_failure_summary() -> AppWorldGate0Summary:
    accounting = _SmokeAccounting()
    accounting.add_diagnostic("smoke.preflight_failure")
    return accounting.summary(seed=100, workers=1)


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
        self._seen_worlds: list[object] = []

    @contextmanager
    def open(self, *, task_id: str, role: str) -> Iterator[object]:
        if type(task_id) is not str or not task_id or role not in _SMOKE_ROLES:
            raise ValueError(_WORLD_FAILURE)
        if not _SMOKE_WORLD_SLOT.acquire(blocking=False):
            self._accounting.execution_exception_count += 1
            raise ValueError(_WORLD_FAILURE) from None
        entered = False
        body_failed = False
        body_completed = False
        pre_yield_failed = False
        yield_started = False
        try:
            try:
                sequence = self._context_count
                self._context_count += 1
                kwargs: dict[str, object] = {
                    "task_id": task_id,
                    "experiment_name": f"toolshift-m3a-{sequence:06d}-{uuid.uuid4().hex}",
                    "random_seed": self._seed,
                    **_WORLD_FLAGS,
                }
                if role == "screen":
                    kwargs["ground_truth_mode"] = "full"
                with self._factory(**kwargs) as world:
                    entered = True
                    if any(world is previous for previous in self._seen_worlds):
                        pre_yield_failed = True
                        raise ValueError(_WORLD_FAILURE)
                    self._seen_worlds.append(world)
                    try:
                        yield_started = True
                        yield world
                    except BaseException:
                        body_failed = True
                        raise
                    else:
                        body_completed = True
                if pre_yield_failed or body_failed:
                    raise ValueError(_WORLD_FAILURE) from None
            except BaseException as error:
                cleanup_failed = _has_cleanup_marker(error) or (
                    entered and body_completed and not body_failed
                )
                if cleanup_failed:
                    self._accounting.cleanup_exception_count += 1
                if body_failed or not yield_started or not cleanup_failed or not entered:
                    self._accounting.execution_exception_count += 1
                raise
        finally:
            _SMOKE_WORLD_SLOT.release()


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


def _open_private_appworld_directory(
    name: str,
    *,
    parent_fd: int,
    create: bool,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    created = False
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
            if created:
                os.chmod(name, 0o700, dir_fd=parent_fd, follow_symlinks=False)
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(parent_fd)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise ValueError
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _read_private_appworld_version(directory_fd: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("version.txt", flags, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise ValueError
        value = os.read(descriptor, 64)
        if os.read(descriptor, 1):
            raise ValueError
        return value.decode("utf-8", errors="strict").strip()
    finally:
        os.close(descriptor)


def _require_private_appworld_tree(root: Path) -> None:
    """Bind protected AppWorld paths without following mutable child symlinks."""

    root_fd: int | None = None
    data_fd: int | None = None
    base_dbs_fd: int | None = None
    runtime_fds: list[int] = []
    try:
        if getattr(os, "O_NOFOLLOW", 0) == 0 or getattr(os, "O_DIRECTORY", 0) == 0:
            raise ValueError
        root_metadata = root.lstat()
        if (
            not root.is_absolute()
            or root.resolve(strict=True) != root
            or stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
        ):
            raise ValueError
        root_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        root_fd = os.open(root, root_flags)
        opened_root = os.fstat(root_fd)
        if (opened_root.st_dev, opened_root.st_ino) != (
            root_metadata.st_dev,
            root_metadata.st_ino,
        ):
            raise ValueError
        data_fd = _open_private_appworld_directory("data", parent_fd=root_fd, create=False)
        base_dbs_fd = _open_private_appworld_directory(
            "base_dbs",
            parent_fd=data_fd,
            create=False,
        )
        if (
            _read_private_appworld_version(data_fd) != _PINNED_APPWORLD_DATA_VERSION
            or _read_private_appworld_version(base_dbs_fd) != _PINNED_APPWORLD_DB_VERSION
        ):
            raise ValueError
        experiments_fd = _open_private_appworld_directory(
            "experiments",
            parent_fd=root_fd,
            create=True,
        )
        runtime_fds.append(experiments_fd)
        runtime_fds.append(
            _open_private_appworld_directory(
                "outputs",
                parent_fd=experiments_fd,
                create=True,
            )
        )
        for name in (".cache", ".tmp", ".profiling", "plots", ".release"):
            runtime_fds.append(
                _open_private_appworld_directory(name, parent_fd=root_fd, create=True)
            )
        current_root = root.lstat()
        if (current_root.st_dev, current_root.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise ValueError
    except BaseException:
        raise RuntimeError(_PREFLIGHT_FAILURE) from None
    finally:
        for descriptor in reversed(runtime_fds):
            os.close(descriptor)
        if base_dbs_fd is not None:
            os.close(base_dbs_fd)
        if data_fd is not None:
            os.close(data_fd)
        if root_fd is not None:
            os.close(root_fd)


def _git_blob_sha1(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


def _parse_lfs_pointer(payload: bytes) -> tuple[str, int] | None:
    lines = payload.splitlines(keepends=True)
    if (
        len(lines) != 3
        or lines[0] != b"version https://git-lfs.github.com/spec/v1\n"
        or not lines[1].startswith(b"oid sha256:")
        or not lines[1].endswith(b"\n")
        or not lines[2].startswith(b"size ")
        or not lines[2].endswith(b"\n")
    ):
        return None
    raw_digest = lines[1][len(b"oid sha256:") : -1]
    raw_size = lines[2][len(b"size ") : -1]
    if (
        len(raw_digest) != 64
        or any(byte not in b"0123456789abcdef" for byte in raw_digest)
        or not raw_size
        or any(byte not in b"0123456789" for byte in raw_size)
        or (len(raw_size) > 1 and raw_size.startswith(b"0"))
    ):
        return None
    return raw_digest.decode("ascii"), int(raw_size)


def _read_head_blobs(
    git_prefix: list[str],
    git_environment: dict[str, str],
) -> tuple[tuple[bytes, bytes, str, bytes], ...]:
    tree = subprocess.run(
        [*git_prefix, "ls-tree", "-r", "-z", "--full-tree", "HEAD"],
        check=True,
        capture_output=True,
        env=git_environment,
        text=False,
        timeout=30,
    )
    entries: list[tuple[bytes, bytes, str]] = []
    seen_paths: set[bytes] = set()
    for record in tree.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        path_parts = raw_path.split(b"/")
        if (
            separator != b"\t"
            or len(fields) != 3
            or fields[0] not in (b"100644", b"100755", b"120000")
            or fields[1] != b"blob"
            or len(fields[2]) != 40
            or any(byte not in b"0123456789abcdef" for byte in fields[2])
            or not raw_path
            or raw_path.startswith(b"/")
            or any(part in (b"", b".", b"..") for part in path_parts)
            or raw_path in seen_paths
        ):
            raise ValueError
        seen_paths.add(raw_path)
        entries.append((fields[0], fields[2], os.fsdecode(raw_path)))
    if not entries:
        raise ValueError

    ordered_oids = tuple(dict.fromkeys(oid for _, oid, _ in entries))
    objects = subprocess.run(
        [*git_prefix, "cat-file", "--batch"],
        input=b"".join(oid + b"\n" for oid in ordered_oids),
        check=True,
        capture_output=True,
        env=git_environment,
        text=False,
        timeout=30,
    ).stdout
    payloads: dict[bytes, bytes] = {}
    offset = 0
    for expected_oid in ordered_oids:
        header_end = objects.find(b"\n", offset)
        if header_end < 0:
            raise ValueError
        header = objects[offset:header_end].split(b" ")
        if (
            len(header) != 3
            or header[0] != expected_oid
            or header[1] != b"blob"
            or not header[2]
            or any(byte not in b"0123456789" for byte in header[2])
        ):
            raise ValueError
        size = int(header[2])
        payload_start = header_end + 1
        payload_end = payload_start + size
        if payload_end >= len(objects) or objects[payload_end : payload_end + 1] != b"\n":
            raise ValueError
        payload = objects[payload_start:payload_end]
        if _git_blob_sha1(payload).encode("ascii") != expected_oid:
            raise ValueError
        payloads[expected_oid] = payload
        offset = payload_end + 1
    if offset != len(objects):
        raise ValueError
    return tuple((mode, oid, path, payloads[oid]) for mode, oid, path in entries)


def _open_checkout_parent(root_fd: int, parts: tuple[bytes, ...]) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in parts:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _trusted_appworld_bundle_secrets(constants_payload: bytes) -> tuple[str, bytes]:
    module = ast.parse(constants_payload.decode("utf-8", errors="strict"))
    literals: dict[str, object] = {}
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in ("PASSWORD", "SALT")
        ):
            name = statement.targets[0].id
            if name in literals:
                raise ValueError
            literals[name] = ast.literal_eval(statement.value)
    bundle_phrase = literals.get("PASSWORD")
    kdf_salt = literals.get("SALT")
    if type(bundle_phrase) is not str or type(kdf_salt) is not bytes:
        raise ValueError
    return bundle_phrase, kdf_salt


def _read_checkout_regular(root_fd: int, raw_path: bytes) -> bytes:
    parts = tuple(raw_path.split(b"/"))
    if (
        not raw_path
        or raw_path.startswith(b"/")
        or any(part in (b"", b".", b"..") for part in parts)
    ):
        raise ValueError
    parent_fd = _open_checkout_parent(root_fd, parts[:-1])
    try:
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
                raise ValueError
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ):
                raise ValueError
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def _decrypt_pinned_appworld_bundles(
    root_fd: int,
    head_payloads: Mapping[bytes, bytes],
) -> dict[bytes, bytes]:
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        bundle_phrase, kdf_salt = _trusted_appworld_bundle_secrets(
            head_payloads[b"src/appworld/common/constants.py"]
        )
        expected_files: dict[bytes, bytes] = {}
        for bundle_path, base_directory in _PINNED_APPWORLD_BUNDLE_LAYOUT:
            pointer = _parse_lfs_pointer(head_payloads[bundle_path])
            if pointer is None:
                raise ValueError
            expected_digest, expected_size = pointer
            encrypted = _read_checkout_regular(root_fd, bundle_path)
            if (
                len(encrypted) != expected_size
                or hashlib.sha256(encrypted).hexdigest() != expected_digest
                or len(encrypted) < 17
            ):
                raise ValueError
            key = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=kdf_salt,
                iterations=100000,
                backend=default_backend(),
            ).derive(bundle_phrase.encode("utf-8"))
            decryptor = Cipher(
                algorithms.AES(key),
                modes.CFB(encrypted[:16]),
                backend=default_backend(),
            ).decryptor()
            archive_bytes = decryptor.update(encrypted[16:]) + decryptor.finalize()
            with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
                for info in archive.infolist():
                    raw_member = info.filename.encode("utf-8", errors="strict")
                    parts = raw_member.split(b"/")
                    if (
                        info.is_dir()
                        or not raw_member
                        or raw_member.startswith(b"/")
                        or b"\\" in raw_member
                        or any(part in (b"", b".", b"..") for part in parts)
                    ):
                        raise ValueError
                    target = base_directory + b"/" + raw_member
                    if target in expected_files or target in head_payloads:
                        raise ValueError
                    expected_files[target] = archive.read(info)
        return expected_files
    except BaseException:
        raise ValueError from None


def _require_untracked_source_layout(
    root_fd: int,
    git_prefix: list[str],
    git_environment: dict[str, str],
    entries: tuple[tuple[bytes, bytes, str, bytes], ...],
) -> None:
    head_payloads = {os.fsencode(path): payload for _, _, path, payload in entries}
    untracked = subprocess.run(
        [*git_prefix, "ls-files", "--others", "-z", "--"],
        check=True,
        capture_output=True,
        env=git_environment,
        text=False,
        timeout=30,
    ).stdout
    actual_paths: set[bytes] = set()
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        if (
            raw_path.startswith(b"/")
            or any(part in (b"", b".", b"..") for part in raw_path.split(b"/"))
            or raw_path in actual_paths
        ):
            raise ValueError
        actual_paths.add(raw_path)
    has_pinned_bundles = all(
        bundle_path in head_payloads for bundle_path, _ in _PINNED_APPWORLD_BUNDLE_LAYOUT
    )
    if not has_pinned_bundles:
        if actual_paths:
            raise ValueError
        return
    expected_files = _decrypt_pinned_appworld_bundles(root_fd, head_payloads)
    if actual_paths != set(expected_files):
        raise ValueError
    for raw_path, expected in expected_files.items():
        if _read_checkout_regular(root_fd, raw_path) != expected:
            raise ValueError


def _require_head_worktree_match(
    checkout: Path,
    git_prefix: list[str],
    git_environment: dict[str, str],
) -> None:
    entries = _read_head_blobs(git_prefix, git_environment)
    root_fd = os.open(checkout, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened_root = os.fstat(root_fd)
    try:
        for mode, oid, relative, expected_payload in entries:
            raw_parts = tuple(os.fsencode(part) for part in Path(relative).parts)
            parent_fd = _open_checkout_parent(root_fd, raw_parts[:-1])
            try:
                leaf = raw_parts[-1]
                if mode == b"120000":
                    metadata = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                    if not stat.S_ISLNK(metadata.st_mode):
                        raise ValueError
                    target = os.readlink(leaf, dir_fd=parent_fd)
                    target_bytes = target if type(target) is bytes else os.fsencode(target)
                    if _git_blob_sha1(target_bytes) != oid.decode("ascii"):
                        raise ValueError
                    continue

                descriptor = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
                try:
                    before = os.fstat(descriptor)
                    if not stat.S_ISREG(before.st_mode) or bool(before.st_mode & 0o111) != (
                        mode == b"100755"
                    ):
                        raise ValueError
                    pointer = _parse_lfs_pointer(expected_payload)
                    if pointer is None:
                        digest = hashlib.sha1(usedforsecurity=False)
                        digest.update(f"blob {before.st_size}\0".encode())
                        expected_size = len(expected_payload)
                    else:
                        expected_digest, expected_size = pointer
                        digest = hashlib.sha256()
                    if before.st_size != expected_size:
                        raise ValueError
                    while chunk := os.read(descriptor, 1024 * 1024):
                        digest.update(chunk)
                    actual_digest = digest.hexdigest()
                    if actual_digest != (
                        oid.decode("ascii") if pointer is None else expected_digest
                    ):
                        raise ValueError
                    after = os.fstat(descriptor)
                    if (
                        after.st_dev,
                        after.st_ino,
                        after.st_mode,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ) != (
                        before.st_dev,
                        before.st_ino,
                        before.st_mode,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    ):
                        raise ValueError
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent_fd)
        _require_untracked_source_layout(root_fd, git_prefix, git_environment, entries)
        current_root = checkout.lstat()
        if (current_root.st_dev, current_root.st_ino) != (
            opened_root.st_dev,
            opened_root.st_ino,
        ):
            raise ValueError
    finally:
        os.close(root_fd)


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
        git_directory = checkout / ".git"
        git_directory_stat = git_directory.lstat()
        if (
            not checkout.is_absolute()
            or stat.S_ISLNK(git_directory_stat.st_mode)
            or not stat.S_ISDIR(git_directory_stat.st_mode)
        ):
            raise ValueError
        git_environment = {
            key: value for key, value in os.environ.items() if not key.startswith("GIT_")
        }
        git_prefix = [
            "git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            f"--git-dir={git_directory}",
            f"--work-tree={checkout}",
        ]
        binding = subprocess.run(
            [*git_prefix, "rev-parse", "--show-toplevel", "--absolute-git-dir"],
            check=True,
            capture_output=True,
            env=git_environment,
            text=True,
            timeout=10,
        )
        binding_lines = binding.stdout.splitlines()
        if (
            len(binding_lines) != 2
            or Path(binding_lines[0]).resolve(strict=True) != checkout
            or Path(binding_lines[1]).resolve(strict=True) != git_directory
        ):
            raise ValueError
        completed = subprocess.run(
            [*git_prefix, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            env=git_environment,
            text=True,
            timeout=10,
        )
        if completed.stdout.strip() != PINNED_APPWORLD_COMMIT:
            raise ValueError
        symbolic = subprocess.run(
            [*git_prefix, "symbolic-ref", "-q", "HEAD"],
            check=False,
            capture_output=True,
            env=git_environment,
            text=True,
            timeout=10,
        )
        if symbolic.returncode != 1:
            raise ValueError
        object_format = subprocess.run(
            [*git_prefix, "rev-parse", "--show-object-format"],
            check=True,
            capture_output=True,
            env=git_environment,
            text=True,
            timeout=10,
        )
        if object_format.stdout.strip() != "sha1":
            raise ValueError
        index_flags = subprocess.run(
            [*git_prefix, "ls-files", "-v", "-z"],
            check=True,
            capture_output=True,
            env=git_environment,
            text=True,
            timeout=10,
        )
        for entry in index_flags.stdout.split("\0"):
            if (
                entry
                and len(entry) >= 2
                and entry[1] == " "
                and (entry[0] == "S" or entry[0].islower())
            ):
                raise ValueError
        staged = subprocess.run(
            [
                *git_prefix,
                "diff-index",
                "--cached",
                "--quiet",
                "--no-ext-diff",
                "--ignore-submodules=none",
                "HEAD",
                "--",
            ],
            check=False,
            capture_output=True,
            env=git_environment,
            text=True,
            timeout=30,
        )
        if staged.returncode != 0 or os.path.lexists(git_directory / "info" / "attributes"):
            raise ValueError
        _require_head_worktree_match(checkout, git_prefix, git_environment)
        return checkout
    except BaseException:
        raise RuntimeError(_PREFLIGHT_FAILURE) from None


def _require_pinned_dotenv_runtime() -> None:
    probe_key = "TOOLSHIFT_DOTENV_DISABLED_PROBE"
    try:
        if (
            importlib.metadata.version("python-dotenv") != _PINNED_PYTHON_DOTENV_VERSION
            or probe_key in os.environ
        ):
            raise ValueError
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = getattr(dotenv_module, "load_dotenv", None)
        if not callable(load_dotenv):
            raise ValueError
        try:
            disabled = load_dotenv(
                stream=io.StringIO(f"{probe_key}=mutated\n"),
                override=True,
            )
            if disabled is not False or probe_key in os.environ:
                raise ValueError
        finally:
            os.environ.pop(probe_key, None)
    except BaseException:
        raise ValueError from None


def require_pinned_appworld_runtime() -> None:
    """Validate the private pinned runtime without exposing paths or values."""

    try:
        if (
            any(key.lower().endswith("_proxy") for key in os.environ)
            or _FORBIDDEN_APPWORLD_RUNTIME_ENVIRONMENT.intersection(os.environ)
            or os.environ.get("PYTHON_DOTENV_DISABLED") != "1"
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or sys.dont_write_bytecode is not True
        ):
            raise ValueError
        if tuple(sys.version_info[:3]) != _PINNED_PYTHON_VERSION:
            raise ValueError
        root = _require_private_appworld_root()
        if os.environ.get("APPWORLD_CACHE") != str(root / ".cache"):
            raise ValueError
        _require_private_appworld_tree(root)
        checkout = _require_editable_appworld_checkout()
        _require_pinned_dotenv_runtime()
        package_module = importlib.import_module("appworld")
        constants_module = importlib.import_module("appworld.common.constants")
        if (
            any(key.lower().endswith("_proxy") for key in os.environ)
            or _FORBIDDEN_APPWORLD_RUNTIME_ENVIRONMENT.intersection(os.environ)
            or os.environ.get("APPWORLD_ROOT") != str(root)
            or os.environ.get("APPWORLD_CACHE") != str(root / ".cache")
            or os.environ.get("PYTHON_DOTENV_DISABLED") != "1"
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or sys.dont_write_bytecode is not True
        ):
            raise ValueError
        package_root = (checkout / "src" / "appworld").resolve(strict=True)
        for module in (package_module, constants_module):
            module_file = getattr(module, "__file__", None)
            if type(module_file) is not str:
                raise ValueError
            Path(module_file).resolve(strict=True).relative_to(package_root)

        if (
            getattr(constants_module, "DATA_VERSION", None) != _PINNED_APPWORLD_DATA_VERSION
            or getattr(constants_module, "DB_VERSION", None) != _PINNED_APPWORLD_DB_VERSION
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


def write_appworld_gate0_summary(
    path: str | Path,
    summary: AppWorldGate0Summary,
) -> None:
    """Write one aggregate smoke result only beneath the validated private root."""

    if type(summary) is not AppWorldGate0Summary:
        raise ValueError("AppWorld Gate 0 summary is invalid") from None
    summary._validate_integrity()
    error_message = "output must be under the private AppWorld root"
    try:
        root = _require_private_appworld_root()
        output = Path(path)
        resolved_root = root.resolve(strict=True)
        resolved_output = output.resolve(strict=False)
        if (
            not root.is_absolute()
            or root != resolved_root
            or root.is_symlink()
            or not root.is_dir()
            or stat.S_IMODE(root.stat().st_mode) != 0o700
            or root.stat().st_uid != os.geteuid()
            or any((parent / ".git").exists() for parent in (root, *root.parents))
            or not output.is_absolute()
            or output == root
            or output.is_symlink()
            or not resolved_output.is_relative_to(resolved_root)
        ):
            raise ValueError
    except BaseException:
        raise ValueError(error_message) from None
    try:
        write_private_json_beneath(root, output, summary.to_dict())
    except BaseException:
        raise ValueError("private AppWorld evidence write failed") from None


__all__ = [
    "APPWORLD_ADAPTER_ABI_SHA256",
    "APPWORLD_GATE0_RUNNER_ABI_SHA256",
    "AppWorldGate0Summary",
    "AppWorldTransformSummary",
    "build_schema_probes",
    "require_pinned_appworld_runtime",
    "run_appworld_gate0_smoke",
    "write_appworld_gate0_summary",
]
