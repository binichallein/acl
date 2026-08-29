"""Four-layer contract orchestration and the hard dataset-admission gate."""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import cast

import toolshift.contracts._schema as schema_contracts
import toolshift.contracts.denotation as denotation_contracts
import toolshift.contracts.state as state_contracts
import toolshift.contracts.trace as trace_contracts
from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts._common import (
    _LAYERS,
    ContractDiagnostic,
    LayerContractResult,
    _diagnostic,
    _fingerprint_parts,
    _make_layer_result,
    _require_safe_identifier,
    _require_sha256,
    _validate_layer_result,
)
from toolshift.contracts._schema import _raw_adapter_variant
from toolshift.types import SchemaVariant, canonical_json_bytes

_INVALID_SUITE_SEQUENCE = object()
_SUITE_ATTESTATION = object()


def contract_suite_fingerprint(
    schema_digest: str,
    results: list[LayerContractResult] | tuple[LayerContractResult, ...],
) -> str:
    """Hash exactly one result per layer from an exact built-in list or tuple."""

    schema_digest = _require_sha256(schema_digest, "schema_fingerprint")
    if type(results) not in (list, tuple):
        raise ValueError("results must be an exact built-in list or tuple")
    validated = tuple(_validate_layer_result(result) for result in results)
    by_layer = {result.layer: result for result in validated}
    if len(validated) != len(_LAYERS) or set(by_layer) != set(_LAYERS):
        raise ValueError("results must contain exactly one result for every layer")
    ordered = tuple(by_layer[layer] for layer in _LAYERS)
    return _fingerprint_parts(
        b"toolshift.contract-suite.v1",
        [canonical_json_bytes(schema_digest)]
        + [bytes.fromhex(result._snapshot_fingerprint) for result in ordered],
    )


@dataclass(frozen=True, eq=False)
class ContractSuiteResult:
    """Four ordered layer results plus sealed aggregate and episode provenance."""

    schema_fingerprint: str
    results: tuple[LayerContractResult, ...]
    fingerprint: str = field(init=False)
    _attestation: object | None = field(init=False, repr=False, default=None)
    _sealed_fingerprint: str | None = field(init=False, repr=False, default=None)
    _state_episode_fingerprint: str | None = field(init=False, repr=False, default=None)
    _trace_episode_fingerprint: str | None = field(init=False, repr=False, default=None)
    _admission_snapshot_fingerprint: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_fingerprint",
            _require_sha256(self.schema_fingerprint, "schema_fingerprint"),
        )
        if not isinstance(self.results, (list, tuple)):
            raise ValueError("results must be a list or tuple")
        validated = tuple(_validate_layer_result(result) for result in self.results)
        by_layer = {result.layer: result for result in validated}
        if len(validated) != len(_LAYERS) or set(by_layer) != set(_LAYERS):
            raise ValueError("results must contain exactly one result for every layer")
        ordered = tuple(by_layer[layer] for layer in _LAYERS)
        object.__setattr__(self, "results", ordered)
        object.__setattr__(
            self,
            "fingerprint",
            contract_suite_fingerprint(self.schema_fingerprint, ordered),
        )

    @property
    def passed(self) -> bool:
        """Return true exactly when all validated layer diagnostics are empty."""

        suite = _validate_suite_result(
            self,
            require_attested=self in _EVALUATED_SUITES,
        )
        return all(len(result.diagnostics) == 0 for result in suite.results)


@dataclass(frozen=True, slots=True)
class _SuiteProvenance:
    results: tuple[LayerContractResult, ...]
    result_snapshots: tuple[str, ...]
    schema_fingerprint: str
    aggregate_fingerprint: str
    state_episode_fingerprint: str
    trace_episode_fingerprint: str
    admission_snapshot_fingerprint: str


_EVALUATED_SUITES: weakref.WeakKeyDictionary[ContractSuiteResult, _SuiteProvenance] = (
    weakref.WeakKeyDictionary()
)


def _admission_snapshot_fingerprint(suite: ContractSuiteResult) -> str:
    aggregate_fingerprint = _require_sha256(
        suite.fingerprint,
        "aggregate_fingerprint",
    )
    state_episodes = _require_sha256(
        suite._state_episode_fingerprint,
        "state_episode_fingerprint",
    )
    trace_episodes = _require_sha256(
        suite._trace_episode_fingerprint,
        "trace_episode_fingerprint",
    )
    return _fingerprint_parts(
        b"toolshift.admission-snapshot.v1",
        [
            bytes.fromhex(aggregate_fingerprint),
            bytes.fromhex(state_episodes),
            bytes.fromhex(trace_episodes),
        ],
    )


def _validate_suite_result(value: object, *, require_attested: bool = False) -> ContractSuiteResult:
    if type(value) is not ContractSuiteResult:
        raise ValueError("suite_result must be a ContractSuiteResult")
    suite = cast(ContractSuiteResult, value)
    if type(suite.results) is not tuple:
        raise ValueError("suite results have been mutated")
    current_schema_fingerprint = _require_sha256(
        suite.schema_fingerprint,
        "schema_fingerprint",
    )
    current_aggregate_fingerprint = _require_sha256(
        suite.fingerprint,
        "aggregate_fingerprint",
    )
    for result in suite.results:
        _validate_layer_result(result, require_attested=require_attested)
    if tuple(result.layer for result in suite.results) != _LAYERS:
        raise ValueError("suite result order has been mutated")
    current_result_snapshots = tuple(
        _require_sha256(
            result._snapshot_fingerprint,
            "layer_snapshot_fingerprint",
        )
        for result in suite.results
    )

    provenance = _EVALUATED_SUITES.get(suite) if require_attested else None
    if require_attested and provenance is None:
        raise ValueError("suite result is not evaluator-attested or issued")
    if provenance is not None:
        if type(provenance.results) is not tuple or type(provenance.result_snapshots) is not tuple:
            raise ValueError("suite external provenance has been mutated")
        provenance_result_snapshots = tuple(
            _require_sha256(value, "provenance_layer_snapshot_fingerprint")
            for value in provenance.result_snapshots
        )
        provenance_schema_fingerprint = _require_sha256(
            provenance.schema_fingerprint,
            "provenance_schema_fingerprint",
        )
        provenance_aggregate_fingerprint = _require_sha256(
            provenance.aggregate_fingerprint,
            "provenance_aggregate_fingerprint",
        )
        provenance_state_episodes = _require_sha256(
            provenance.state_episode_fingerprint,
            "provenance_state_episode_fingerprint",
        )
        provenance_trace_episodes = _require_sha256(
            provenance.trace_episode_fingerprint,
            "provenance_trace_episode_fingerprint",
        )
        provenance_admission_snapshot = _require_sha256(
            provenance.admission_snapshot_fingerprint,
            "provenance_admission_snapshot_fingerprint",
        )
        current_state_episodes = _require_sha256(
            suite._state_episode_fingerprint,
            "state_episode_fingerprint",
        )
        current_trace_episodes = _require_sha256(
            suite._trace_episode_fingerprint,
            "trace_episode_fingerprint",
        )
        current_admission_snapshot = _require_sha256(
            suite._admission_snapshot_fingerprint,
            "admission_snapshot_fingerprint",
        )
        if (
            len(suite.results) != len(provenance.results)
            or any(
                current is not issued
                for current, issued in zip(
                    suite.results,
                    provenance.results,
                    strict=True,
                )
            )
            or current_result_snapshots != provenance_result_snapshots
            or current_schema_fingerprint != provenance_schema_fingerprint
            or current_aggregate_fingerprint != provenance_aggregate_fingerprint
            or current_state_episodes != provenance_state_episodes
            or current_trace_episodes != provenance_trace_episodes
            or current_admission_snapshot != provenance_admission_snapshot
        ):
            raise ValueError("suite fingerprint or external provenance has been mutated")

    rebuilt = ContractSuiteResult(current_schema_fingerprint, suite.results)
    if current_aggregate_fingerprint != rebuilt.fingerprint:
        raise ValueError("suite fingerprint does not match its results")
    if require_attested:
        sealed_fingerprint = _require_sha256(
            suite._sealed_fingerprint,
            "sealed_aggregate_fingerprint",
        )
        current_admission_snapshot = _require_sha256(
            suite._admission_snapshot_fingerprint,
            "admission_snapshot_fingerprint",
        )
        if (
            suite._attestation is not _SUITE_ATTESTATION
            or sealed_fingerprint != current_aggregate_fingerprint
        ):
            raise ValueError("suite result is not evaluator-attested")
        if current_admission_snapshot != _admission_snapshot_fingerprint(suite):
            raise ValueError("suite admission provenance has been mutated")
    return suite


def _run_suite_checker(
    layer: str,
    fallback_fingerprint: str,
    checker: Callable[[], LayerContractResult],
) -> LayerContractResult:
    try:
        result = _validate_layer_result(checker(), require_attested=True)
        if result.layer != layer:
            raise ValueError("checker returned the wrong layer")
        return result
    except Exception:
        return _make_layer_result(
            layer,
            fallback_fingerprint,
            0,
            [
                _diagnostic(
                    f"suite.{layer}_exception",
                    f"{layer.title()} layer checker raised",
                )
            ],
        )


def _episode_fingerprint(values: Sequence[str]) -> str:
    return _fingerprint_parts(
        b"toolshift.episode-set.v1",
        [canonical_json_bytes(value) for value in sorted(values)],
    )


def _freeze_suite_sequence(value: object) -> object:
    if type(value) not in (list, tuple):
        return _INVALID_SUITE_SEQUENCE
    try:
        return tuple(value)
    except Exception:
        return _INVALID_SUITE_SEQUENCE


@dataclass(frozen=True, slots=True)
class _EntryCaseSnapshot:
    value: object
    fingerprint: str


@dataclass(frozen=True, slots=True)
class _EntryCaseGroup:
    snapshots: tuple[_EntryCaseSnapshot, ...]
    validator: Callable[[object], object]
    all_valid: bool
    capture_episode_ids: bool
    episode_ids: tuple[str, ...] | None


def _capture_entry_case_group(
    value: object,
    expected_type: type[object],
    shape_validator: Callable[[object], bool],
    validator: Callable[[object], object],
    *,
    capture_episode_ids: bool = False,
) -> _EntryCaseGroup:
    if type(value) is not tuple:
        return _EntryCaseGroup((), validator, False, capture_episode_ids, None)
    snapshots: list[_EntryCaseSnapshot] = []
    all_valid = True
    for raw_case in value:
        if type(raw_case) is not expected_type:
            all_valid = False
            continue
        try:
            if not shape_validator(raw_case):
                raise ValueError("case shape validation failed")
            fingerprint = _require_sha256(
                object.__getattribute__(raw_case, "_snapshot_fingerprint"),
                "case_snapshot_fingerprint",
            )
        except Exception:
            all_valid = False
            continue
        snapshots.append(_EntryCaseSnapshot(raw_case, fingerprint))
    return _EntryCaseGroup(
        tuple(snapshots),
        validator,
        all_valid,
        capture_episode_ids,
        None,
    )


def _validate_entry_case_group(group: _EntryCaseGroup) -> _EntryCaseGroup:
    all_valid = group.all_valid
    episode_ids: list[str] = []
    for snapshot in group.snapshots:
        try:
            validated = group.validator(snapshot.value)
            current_fingerprint = _require_sha256(
                validated._snapshot_fingerprint,
                "case_snapshot_fingerprint",
            )
            if validated is not snapshot.value or current_fingerprint != snapshot.fingerprint:
                raise ValueError("case validation changed entry snapshot")
            if group.capture_episode_ids:
                episode_ids.append(_require_safe_identifier(validated.episode_id, "episode_id"))
        except Exception:
            all_valid = False
    frozen_episode_ids = tuple(episode_ids) if group.capture_episode_ids and all_valid else None
    return _EntryCaseGroup(
        group.snapshots,
        group.validator,
        all_valid,
        group.capture_episode_ids,
        frozen_episode_ids,
    )


def _entry_case_groups_match(groups: Sequence[_EntryCaseGroup]) -> bool:
    matches = True
    for group in groups:
        if not group.all_valid:
            matches = False
        for snapshot in group.snapshots:
            try:
                current = group.validator(snapshot.value)
                current_fingerprint = _require_sha256(
                    current._snapshot_fingerprint,
                    "case_snapshot_fingerprint",
                )
            except Exception:
                matches = False
                continue
            if current is not snapshot.value or current_fingerprint != snapshot.fingerprint:
                matches = False
    return matches


@dataclass(frozen=True, slots=True)
class _FrozenSuiteInputs:
    schema_probes: object
    denotation_cases: object
    state_cases: object
    trace_cases: object


def _freeze_suite_inputs(
    schema_probes: object,
    denotation_cases: object,
    state_cases: object,
    trace_cases: object,
) -> _FrozenSuiteInputs:
    return _FrozenSuiteInputs(
        _freeze_suite_sequence(schema_probes),
        _freeze_suite_sequence(denotation_cases),
        _freeze_suite_sequence(state_cases),
        _freeze_suite_sequence(trace_cases),
    )


@dataclass(frozen=True, slots=True)
class _SuiteEntry:
    case_groups: tuple[_EntryCaseGroup, ...]
    variant_fingerprint: str
    supplied_variant_valid: bool
    adapter_entry_variant: object | None
    adapter_entry_fingerprint: str | None
    adapter_public_entry_fingerprint: str | None


def _capture_suite_entry(
    variant: SchemaVariant,
    adapter: SemanticAdapter,
    inputs: _FrozenSuiteInputs,
) -> _SuiteEntry:
    raw_case_groups = (
        _capture_entry_case_group(
            inputs.schema_probes,
            schema_contracts.SchemaProbe,
            schema_contracts._schema_probe_has_canonical_shape,
            schema_contracts._validate_schema_probe,
        ),
        _capture_entry_case_group(
            inputs.denotation_cases,
            denotation_contracts.DenotationCase,
            denotation_contracts._denotation_case_has_canonical_shape,
            denotation_contracts._validate_denotation_case,
        ),
        _capture_entry_case_group(
            inputs.state_cases,
            state_contracts.StateCase,
            state_contracts._state_case_has_canonical_shape,
            state_contracts._validate_state_case,
            capture_episode_ids=True,
        ),
        _capture_entry_case_group(
            inputs.trace_cases,
            trace_contracts.TraceCase,
            trace_contracts._trace_case_has_canonical_shape,
            trace_contracts._validate_trace_case,
            capture_episode_ids=True,
        ),
    )

    try:
        variant_fingerprint = schema_contracts.schema_fingerprint(variant)
        supplied_variant_valid = True
    except Exception:
        supplied_variant_valid = False
        variant_fingerprint = hashlib.sha256(b"toolshift.schema.invalid").hexdigest()

    try:
        adapter_entry_variant = _raw_adapter_variant(adapter)
        adapter_entry_fingerprint = schema_contracts.schema_fingerprint(adapter_entry_variant)
    except Exception:
        adapter_entry_variant = None
        adapter_entry_fingerprint = None

    try:
        adapter_public_entry_fingerprint = schema_contracts.schema_fingerprint(adapter.variant)
    except Exception:
        adapter_public_entry_fingerprint = None

    return _SuiteEntry(
        tuple(_validate_entry_case_group(group) for group in raw_case_groups),
        variant_fingerprint,
        supplied_variant_valid,
        adapter_entry_variant,
        adapter_entry_fingerprint,
        adapter_public_entry_fingerprint,
    )


@dataclass(slots=True)
class _SuiteGuard:
    variant: SchemaVariant
    adapter: SemanticAdapter
    entry: _SuiteEntry
    case_snapshot_mismatch_detected: bool = False
    variant_binding_mismatch_detected: bool = False

    def _public_binding_matches(self) -> bool:
        try:
            return (
                self.entry.adapter_public_entry_fingerprint == self.entry.variant_fingerprint
                and schema_contracts.schema_fingerprint(self.adapter.variant)
                == self.entry.variant_fingerprint
            )
        except Exception:
            return False

    def _raw_binding_matches(self) -> bool:
        try:
            current_adapter_variant = _raw_adapter_variant(self.adapter)
            return (
                self.entry.supplied_variant_valid
                and self.entry.adapter_entry_fingerprint == self.entry.variant_fingerprint
                and current_adapter_variant is self.entry.adapter_entry_variant
                and schema_contracts.schema_fingerprint(self.variant)
                == self.entry.variant_fingerprint
                and schema_contracts.schema_fingerprint(current_adapter_variant)
                == self.entry.variant_fingerprint
            )
        except Exception:
            return False

    def apply(self, result: LayerContractResult) -> LayerContractResult:
        """Apply public binding, case snapshot, then callback-free raw seal checks."""

        public_binding_matches = self._public_binding_matches()
        case_snapshots_match = _entry_case_groups_match(self.entry.case_groups)
        raw_binding_matches = self._raw_binding_matches()
        if not public_binding_matches or not raw_binding_matches:
            self.variant_binding_mismatch_detected = True
        if not case_snapshots_match:
            self.case_snapshot_mismatch_detected = True

        extra_diagnostics: list[ContractDiagnostic] = []
        if self.case_snapshot_mismatch_detected:
            extra_diagnostics.append(
                _diagnostic(
                    "suite.case_snapshot_mismatch",
                    "Entry case validation or content stability failed",
                )
            )
        if self.variant_binding_mismatch_detected:
            extra_diagnostics.append(
                _diagnostic(
                    "suite.variant_binding_mismatch",
                    "Adapter binding changed during contract evaluation",
                )
            )
        if not extra_diagnostics:
            return result
        return _make_layer_result(
            result.layer,
            result.fingerprint,
            result.checks_run,
            (*result.diagnostics, *extra_diagnostics),
        )


def _run_all_layers(
    variant: SchemaVariant,
    adapter: SemanticAdapter,
    inputs: _FrozenSuiteInputs,
    state_provider: state_contracts.StateEvidenceProvider,
    trace_provider: trace_contracts.TraceEvidenceProvider,
    guard: _SuiteGuard,
) -> tuple[LayerContractResult, ...]:
    schema_result = guard.apply(
        _run_suite_checker(
            "schema",
            guard.entry.variant_fingerprint,
            lambda: schema_contracts.check_schema_contract(
                variant,
                adapter,
                inputs.schema_probes,
            ),
        )
    )
    denotation_result = guard.apply(
        _run_suite_checker(
            "denotation",
            hashlib.sha256(b"toolshift.denotation.checker-error").hexdigest(),
            lambda: denotation_contracts.check_denotation_contract(
                adapter,
                inputs.denotation_cases,
            ),
        )
    )
    state_result = guard.apply(
        _run_suite_checker(
            "state",
            hashlib.sha256(b"toolshift.state.checker-error").hexdigest(),
            lambda: state_contracts.check_state_contract(
                inputs.state_cases,
                state_provider,
            ),
        )
    )
    trace_result = guard.apply(
        _run_suite_checker(
            "trace",
            hashlib.sha256(b"toolshift.trace.checker-error").hexdigest(),
            lambda: trace_contracts.check_trace_contract(
                adapter,
                inputs.trace_cases,
                trace_provider,
                inputs.denotation_cases,
            ),
        )
    )
    if guard.variant_binding_mismatch_detected and not any(
        diagnostic.code == "suite.variant_binding_mismatch"
        for diagnostic in schema_result.diagnostics
    ):
        schema_result = _make_layer_result(
            "schema",
            schema_result.fingerprint,
            schema_result.checks_run,
            (
                *schema_result.diagnostics,
                _diagnostic(
                    "suite.variant_binding_mismatch",
                    "Adapter binding changed during contract evaluation",
                ),
            ),
        )
    return schema_result, denotation_result, state_result, trace_result


def _bind_episode_provenance(
    results: tuple[LayerContractResult, ...],
    entry: _SuiteEntry,
) -> tuple[tuple[LayerContractResult, ...], str, str]:
    state_episode_ids = entry.case_groups[2].episode_ids
    if state_episode_ids is None:
        state_fingerprint = hashlib.sha256(b"toolshift.state-episodes.invalid").hexdigest()
    else:
        state_fingerprint = _episode_fingerprint(state_episode_ids)
    trace_episode_ids = entry.case_groups[3].episode_ids
    if trace_episode_ids is None:
        trace_fingerprint = hashlib.sha256(b"toolshift.trace-episodes.invalid").hexdigest()
    else:
        trace_fingerprint = _episode_fingerprint(trace_episode_ids)

    if state_fingerprint != trace_fingerprint:
        trace_result = results[3]
        trace_result = _make_layer_result(
            "trace",
            trace_result.fingerprint,
            trace_result.checks_run,
            (
                *trace_result.diagnostics,
                _diagnostic(
                    "suite.episode_set_mismatch",
                    "State and trace episode identifier sets differ",
                ),
            ),
        )
        results = (*results[:3], trace_result)
    return results, state_fingerprint, trace_fingerprint


def _seal_suite(
    variant_fingerprint: str,
    results: tuple[LayerContractResult, ...],
    state_episode_fingerprint: str,
    trace_episode_fingerprint: str,
) -> ContractSuiteResult:
    suite = ContractSuiteResult(variant_fingerprint, results)
    object.__setattr__(suite, "_attestation", _SUITE_ATTESTATION)
    object.__setattr__(suite, "_sealed_fingerprint", suite.fingerprint)
    object.__setattr__(
        suite,
        "_state_episode_fingerprint",
        state_episode_fingerprint,
    )
    object.__setattr__(
        suite,
        "_trace_episode_fingerprint",
        trace_episode_fingerprint,
    )
    object.__setattr__(
        suite,
        "_admission_snapshot_fingerprint",
        _admission_snapshot_fingerprint(suite),
    )
    _EVALUATED_SUITES[suite] = _SuiteProvenance(
        results=suite.results,
        result_snapshots=tuple(result._snapshot_fingerprint for result in suite.results),
        schema_fingerprint=suite.schema_fingerprint,
        aggregate_fingerprint=suite.fingerprint,
        state_episode_fingerprint=state_episode_fingerprint,
        trace_episode_fingerprint=trace_episode_fingerprint,
        admission_snapshot_fingerprint=cast(
            str,
            suite._admission_snapshot_fingerprint,
        ),
    )
    return suite


def evaluate_contract_suite(
    variant: SchemaVariant,
    adapter: SemanticAdapter,
    schema_probes: list[schema_contracts.SchemaProbe] | tuple[schema_contracts.SchemaProbe, ...],
    denotation_cases: list[denotation_contracts.DenotationCase]
    | tuple[denotation_contracts.DenotationCase, ...],
    state_cases: list[state_contracts.StateCase] | tuple[state_contracts.StateCase, ...],
    state_provider: state_contracts.StateEvidenceProvider,
    trace_cases: list[trace_contracts.TraceCase] | tuple[trace_contracts.TraceCase, ...],
    trace_provider: trace_contracts.TraceEvidenceProvider,
) -> ContractSuiteResult:
    """Run all layers over exact built-in list or tuple inputs without short-circuiting.

    All four outer case collections are frozen once at entry. Every checker runs
    even when an earlier layer fails. State providers submit opaque whole-state
    and collateral digests; trace providers attach opaque effect digests to the
    supplied traces and scores. Their episode identifier sets must match. The
    returned object carries process-local evaluator provenance required by
    :func:`require_dataset_admission`.
    """

    inputs = _freeze_suite_inputs(
        schema_probes,
        denotation_cases,
        state_cases,
        trace_cases,
    )
    entry = _capture_suite_entry(variant, adapter, inputs)
    guard = _SuiteGuard(variant, adapter, entry)
    results = _run_all_layers(
        variant,
        adapter,
        inputs,
        state_provider,
        trace_provider,
        guard,
    )
    results, state_fingerprint, trace_fingerprint = _bind_episode_provenance(
        results,
        entry,
    )
    return _seal_suite(
        entry.variant_fingerprint,
        results,
        state_fingerprint,
        trace_fingerprint,
    )


def require_dataset_admission(
    variant: SchemaVariant,
    suite_result: ContractSuiteResult,
) -> ContractSuiteResult:
    """Return only a sealed, non-vacuous, passing four-layer suite for ``variant``.

    The gate revalidates all results, fingerprints, process-local evaluator
    provenance, and paired state/trace episode evidence; every mismatch raises
    ``ValueError``. Admission must run in the same process on the same object
    returned by :func:`evaluate_contract_suite`; serialization or reconstruction
    cannot preserve the private attestation.
    """

    expected_schema_fingerprint = schema_contracts.schema_fingerprint(variant)
    suite = _validate_suite_result(suite_result, require_attested=True)
    if suite.schema_fingerprint != expected_schema_fingerprint:
        raise ValueError("suite schema fingerprint does not match variant")
    schema_result = suite.results[0]
    if schema_result.layer != "schema" or schema_result.fingerprint != expected_schema_fingerprint:
        raise ValueError("schema layer fingerprint does not match variant")
    if any(result.checks_run <= 0 for result in suite.results):
        raise ValueError("contract evidence must be non-vacuous for every layer")
    if any(len(result.diagnostics) != 0 for result in suite.results):
        raise ValueError("all contract layers must pass before dataset admission")
    if (
        suite._state_episode_fingerprint is None
        or suite._trace_episode_fingerprint is None
        or suite._state_episode_fingerprint != suite._trace_episode_fingerprint
    ):
        raise ValueError("state and trace episode identifier sets must match")
    if contract_suite_fingerprint(suite.schema_fingerprint, suite.results) != suite.fingerprint:
        raise ValueError("suite aggregate fingerprint validation failed")
    return suite


__all__ = [
    "ContractSuiteResult",
    "contract_suite_fingerprint",
    "evaluate_contract_suite",
    "require_dataset_admission",
]
