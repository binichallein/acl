"""Schema-level behavioral-equivalence contracts."""

from __future__ import annotations

import hashlib
import re
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.types import (
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    _freeze_mapping,
    canonical_json_bytes,
)

_LAYERS = ("schema", "denotation", "state", "trace")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}\Z")
_SAFE_MESSAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,;:()'=_+-]{0,239}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LAYER_ATTESTATION = object()
_SUITE_ATTESTATION = object()
_INVALID_SUITE_SEQUENCE = object()


def _require_safe_identifier(value: object, context: str) -> str:
    if type(value) is not str or _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{context} must be a non-blank payload-safe identifier")
    return value


def _require_safe_message(value: object) -> str:
    if type(value) is not str or _SAFE_MESSAGE.fullmatch(value) is None:
        raise ValueError("message must be non-blank payload-safe text")
    return value


def _require_sha256(value: object, context: str = "fingerprint") -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return value


def _require_non_blank_text(value: object, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{context} must be a non-blank string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must be valid UTF-8 text") from error
    return value


def _fingerprint_parts(domain: bytes, parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for part in parts:
        _update_framed(digest, part)
    return digest.hexdigest()


def _snapshot_action(value: object, context: str) -> SemanticAction:
    if type(value) is not SemanticAction:
        raise ValueError(f"{context} must contain only SemanticAction values")
    action = cast(SemanticAction, value)
    rebuilt = SemanticAction(action.name, action.arguments)
    if action != rebuilt:
        raise ValueError(f"{context} contains a mutated SemanticAction")
    return rebuilt


def _snapshot_actions(value: object, context: str) -> tuple[SemanticAction, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list or tuple")
    return tuple(_snapshot_action(action, context) for action in value)


def _snapshot_call(value: object, context: str) -> Mapping[str, JSONValue]:
    return _freeze_mapping(value, context)


def _call_fingerprint(call: Mapping[str, JSONValue]) -> str:
    return hashlib.sha256(canonical_json_bytes(call)).hexdigest()


def _actions_equal(
    first: tuple[SemanticAction, ...], second: tuple[SemanticAction, ...]
) -> bool:
    return len(first) == len(second) and all(
        left == right for left, right in zip(first, second, strict=True)
    )


def _validate_schema_variant(variant: object) -> SchemaVariant:
    if type(variant) is not SchemaVariant:
        raise ValueError("variant must be a SchemaVariant")
    typed = cast(SchemaVariant, variant)
    if type(typed.tools) is not tuple:
        raise ValueError("variant contains mutated tools")
    rebuilt_tools: list[SurfaceToolSpec] = []
    for tool in typed.tools:
        if type(tool) is not SurfaceToolSpec:
            raise ValueError("variant tools must contain only SurfaceToolSpec values")
        rebuilt_tool = SurfaceToolSpec(tool.name, tool.description, tool.input_schema)
        if tool != rebuilt_tool:
            raise ValueError("variant contains a mutated tool")
        rebuilt_tools.append(rebuilt_tool)
    rebuilt = SchemaVariant(typed.variant_id, rebuilt_tools, typed.manifest)
    if typed != rebuilt:
        raise ValueError("variant has been mutated")
    return rebuilt


@dataclass(frozen=True, slots=True)
class ContractDiagnostic:
    """A payload-free description of one failed contract check."""

    code: str
    message: str
    case_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_safe_identifier(self.code, "code"))
        object.__setattr__(self, "message", _require_safe_message(self.message))
        object.__setattr__(
            self,
            "case_id",
            _require_safe_identifier(self.case_id, "case_id"),
        )


@dataclass(frozen=True, slots=True)
class LayerContractResult:
    """Immutable diagnostics and evidence fingerprint for one contract layer."""

    layer: str
    fingerprint: str
    checks_run: int
    diagnostics: tuple[ContractDiagnostic, ...]
    _snapshot_fingerprint: str = field(init=False, repr=False)
    _attestation: object | None = field(init=False, repr=False, default=None)
    _sealed_snapshot_fingerprint: str | None = field(
        init=False, repr=False, default=None
    )
    LAYERS: ClassVar[tuple[str, ...]] = _LAYERS

    def __post_init__(self) -> None:
        if type(self.layer) is not str or self.layer not in self.LAYERS:
            raise ValueError("layer must be schema, denotation, state, or trace")
        object.__setattr__(self, "fingerprint", _require_sha256(self.fingerprint))
        if type(self.checks_run) is not int:
            raise ValueError("checks_run must be a nonnegative integer")
        if self.checks_run < 0:
            raise ValueError("checks_run must be a nonnegative integer")
        if not isinstance(self.diagnostics, (list, tuple)):
            raise ValueError("diagnostics must be a list or tuple")
        diagnostics = tuple(_validate_diagnostic(item) for item in self.diagnostics)
        object.__setattr__(self, "diagnostics", diagnostics)
        parts = [
            canonical_json_bytes(self.layer),
            canonical_json_bytes(self.fingerprint),
            canonical_json_bytes(self.checks_run),
            canonical_json_bytes(len(diagnostics)),
        ]
        for diagnostic in diagnostics:
            parts.extend(
                (
                    canonical_json_bytes(diagnostic.code),
                    canonical_json_bytes(diagnostic.message),
                    canonical_json_bytes(diagnostic.case_id),
                )
            )
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            _fingerprint_parts(b"toolshift.layer-result.v1", parts),
        )

    @property
    def passed(self) -> bool:
        """Return true exactly when no diagnostics were recorded."""

        result = _validate_layer_result(self)
        return len(result.diagnostics) == 0


def _validate_diagnostic(value: object) -> ContractDiagnostic:
    if type(value) is not ContractDiagnostic:
        raise ValueError("diagnostics must contain only ContractDiagnostic values")
    diagnostic = cast(ContractDiagnostic, value)
    rebuilt = ContractDiagnostic(diagnostic.code, diagnostic.message, diagnostic.case_id)
    if diagnostic != rebuilt:
        raise ValueError("diagnostic has been mutated")
    return diagnostic


def _validate_layer_result(
    value: object, *, require_attested: bool = False
) -> LayerContractResult:
    if type(value) is not LayerContractResult:
        raise ValueError("results must contain only LayerContractResult values")
    result = cast(LayerContractResult, value)
    if type(result.diagnostics) is not tuple:
        raise ValueError("layer result diagnostics have been mutated")
    snapshot_fingerprint = _require_sha256(
        result._snapshot_fingerprint,
        "layer_snapshot_fingerprint",
    )
    diagnostics = tuple(_validate_diagnostic(item) for item in result.diagnostics)
    rebuilt = LayerContractResult(
        result.layer,
        result.fingerprint,
        result.checks_run,
        diagnostics,
    )
    if snapshot_fingerprint != rebuilt._snapshot_fingerprint:
        raise ValueError("layer result has been mutated")
    if require_attested:
        if result._attestation is not _LAYER_ATTESTATION:
            raise ValueError("layer result is not checker-attested")
        sealed_fingerprint = _require_sha256(
            result._sealed_snapshot_fingerprint,
            "sealed_layer_snapshot_fingerprint",
        )
        if sealed_fingerprint != snapshot_fingerprint:
            raise ValueError("layer result is not checker-attested")
    return result


def _make_layer_result(
    layer: str,
    fingerprint: str,
    checks_run: int,
    diagnostics: Sequence[ContractDiagnostic],
) -> LayerContractResult:
    result = LayerContractResult(layer, fingerprint, checks_run, tuple(diagnostics))
    object.__setattr__(result, "_attestation", _LAYER_ATTESTATION)
    object.__setattr__(
        result,
        "_sealed_snapshot_fingerprint",
        result._snapshot_fingerprint,
    )
    return result


def contract_suite_fingerprint(
    schema_digest: str,
    results: Sequence[LayerContractResult],
) -> str:
    """Recompute the aggregate fingerprint for exactly four layer results."""

    schema_digest = _require_sha256(schema_digest, "schema_fingerprint")
    if not isinstance(results, (list, tuple)):
        raise ValueError("results must be a list or tuple")
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
    """Exactly four layer results plus recomputable aggregate provenance."""

    schema_fingerprint: str
    results: tuple[LayerContractResult, ...]
    fingerprint: str = field(init=False)
    _attestation: object | None = field(init=False, repr=False, default=None)
    _sealed_fingerprint: str | None = field(init=False, repr=False, default=None)
    _state_episode_fingerprint: str | None = field(init=False, repr=False, default=None)
    _trace_episode_fingerprint: str | None = field(init=False, repr=False, default=None)
    _admission_snapshot_fingerprint: str | None = field(
        init=False, repr=False, default=None
    )

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
        """Return true exactly when every layer has zero diagnostics."""

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


_EVALUATED_SUITES: weakref.WeakKeyDictionary[
    ContractSuiteResult, _SuiteProvenance
] = weakref.WeakKeyDictionary()


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


@dataclass(frozen=True, slots=True, eq=False)
class SchemaProbe:
    """One expected surface-to-semantic schema mapping."""

    case_id: str
    surface_tool_name: str
    surface_call: Mapping[str, JSONValue]
    expected_actions: tuple[SemanticAction, ...]
    _snapshot_fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _require_safe_identifier(self.case_id, "case_id"),
        )
        object.__setattr__(
            self,
            "surface_tool_name",
            _require_non_blank_text(self.surface_tool_name, "surface_tool_name"),
        )
        surface_call = _snapshot_call(self.surface_call, "surface_call")
        expected_actions = _snapshot_actions(self.expected_actions, "expected_actions")
        object.__setattr__(self, "surface_call", surface_call)
        object.__setattr__(self, "expected_actions", expected_actions)
        parts = [
            canonical_json_bytes(self.case_id),
            canonical_json_bytes(self.surface_tool_name),
            canonical_json_bytes(surface_call),
        ]
        for action in expected_actions:
            parts.extend(
                (canonical_json_bytes(action.name), canonical_json_bytes(action.arguments))
            )
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            _fingerprint_parts(b"toolshift.schema-probe.v1", parts),
        )


def _validate_schema_probe(value: object) -> SchemaProbe:
    if type(value) is not SchemaProbe:
        raise ValueError("probes must contain only SchemaProbe values")
    probe = cast(SchemaProbe, value)
    rebuilt = SchemaProbe(
        probe.case_id,
        probe.surface_tool_name,
        probe.surface_call,
        probe.expected_actions,
    )
    snapshot_fingerprint = _require_sha256(
        probe._snapshot_fingerprint,
        "schema_probe_snapshot_fingerprint",
    )
    if snapshot_fingerprint != rebuilt._snapshot_fingerprint:
        raise ValueError("probe has been mutated")
    return probe


def _diagnostic(code: str, message: str, case_id: str = "suite") -> ContractDiagnostic:
    return ContractDiagnostic(code=code, message=message, case_id=case_id)


def _update_framed(digest: object, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def schema_fingerprint(variant: SchemaVariant) -> str:
    """Hash a variant identifier, ordered full tool specs, and manifest."""

    variant = _validate_schema_variant(variant)
    digest = hashlib.sha256(b"toolshift.schema.v1\0")
    _update_framed(digest, canonical_json_bytes(variant.variant_id))
    for tool in variant.tools:
        _update_framed(digest, canonical_json_bytes(tool.name))
        _update_framed(digest, canonical_json_bytes(tool.description))
        _update_framed(digest, canonical_json_bytes(tool.input_schema))
    _update_framed(digest, canonical_json_bytes(variant.manifest))
    return digest.hexdigest()


def check_schema_contract(
    variant: SchemaVariant,
    adapter: SemanticAdapter,
    probes: Sequence[SchemaProbe],
) -> LayerContractResult:
    """Verify complete deterministic surface-to-semantic schema mappings."""

    diagnostics: list[ContractDiagnostic] = []
    try:
        fingerprint = schema_fingerprint(variant)
        validated_variant = _validate_schema_variant(variant)
    except Exception:
        fingerprint = hashlib.sha256(b"toolshift.schema.invalid").hexdigest()
        validated_variant = None
        diagnostics.append(
            _diagnostic("schema.invalid_variant", "Schema variant validation failed")
        )

    try:
        bound_fingerprint = schema_fingerprint(adapter.variant)
        if validated_variant is None or bound_fingerprint != fingerprint:
            diagnostics.append(
                _diagnostic("schema.variant_mismatch", "Adapter variant does not match schema")
            )
    except Exception:
        diagnostics.append(
            _diagnostic("schema.variant_exception", "Adapter variant validation raised")
        )

    def adapter_binding_matches() -> bool:
        if validated_variant is None:
            return False
        try:
            return (
                schema_fingerprint(variant) == fingerprint
                and schema_fingerprint(adapter.variant) == fingerprint
            )
        except Exception:
            return False

    if not isinstance(probes, (list, tuple)):
        diagnostics.append(
            _diagnostic("schema.invalid_probes", "Schema probes must be a finite sequence")
        )
        probe_values: tuple[object, ...] = ()
    else:
        probe_values = tuple(probes)
    if not probe_values:
        diagnostics.append(
            _diagnostic("schema.empty_probes", "Schema probes must not be empty")
        )

    valid_probes: list[SchemaProbe] = []
    for raw_probe in probe_values:
        try:
            valid_probes.append(_validate_schema_probe(raw_probe))
        except Exception:
            case_id = getattr(raw_probe, "case_id", "suite")
            if type(case_id) is not str or _SAFE_IDENTIFIER.fullmatch(case_id) is None:
                case_id = "suite"
            diagnostics.append(
                _diagnostic("schema.invalid_probe", "Schema probe validation failed", case_id)
            )

    case_ids = [probe.case_id for probe in valid_probes]
    if len(case_ids) != len(set(case_ids)):
        diagnostics.append(
            _diagnostic("schema.duplicate_case", "Schema case identifiers must be unique")
        )
    call_fingerprints = [_call_fingerprint(probe.surface_call) for probe in valid_probes]
    if len(call_fingerprints) != len(set(call_fingerprints)):
        diagnostics.append(
            _diagnostic("schema.duplicate_call", "Schema surface calls must be unique")
        )

    declared_names = (
        {tool.name for tool in validated_variant.tools}
        if validated_variant is not None
        else set()
    )
    probed_names = {probe.surface_tool_name for probe in valid_probes}
    if declared_names - probed_names:
        diagnostics.append(
            _diagnostic("schema.missing_tool", "Declared surface tools require probes")
        )
    if probed_names - declared_names:
        diagnostics.append(
            _diagnostic("schema.undeclared_tool", "Probes include undeclared surface tools")
        )

    for probe in valid_probes:
        if probe.surface_call.get("name") != probe.surface_tool_name:
            diagnostics.append(
                _diagnostic(
                    "schema.call_name_mismatch",
                    "Surface call name does not match probe tool",
                    probe.case_id,
                )
            )
        parsed: list[tuple[SemanticAction, ...]] = []
        raised = False
        variant_rebound = False
        for _ in range(2):
            try:
                output = adapter.surface_to_semantic(probe.surface_call)
                if not isinstance(output, tuple):
                    raise ValueError("adapter output must be a tuple")
                parsed.append(_snapshot_actions(output, "adapter actions"))
            except Exception:
                raised = True
            finally:
                if not adapter_binding_matches():
                    variant_rebound = True
        if variant_rebound:
            diagnostics.append(
                _diagnostic(
                    "schema.variant_rebound",
                    "Adapter variant changed during surface mapping",
                    probe.case_id,
                )
            )
        if raised or len(parsed) != 2:
            diagnostics.append(
                _diagnostic(
                    "schema.mapping_exception",
                    "Surface mapping raised or returned malformed actions",
                    probe.case_id,
                )
            )
            continue
        if not _actions_equal(parsed[0], parsed[1]):
            diagnostics.append(
                _diagnostic(
                    "schema.nondeterministic_mapping",
                    "Surface mapping is not deterministic",
                    probe.case_id,
                )
            )
        if not _actions_equal(parsed[0], probe.expected_actions):
            diagnostics.append(
                _diagnostic(
                    "schema.mapping_mismatch",
                    "Surface mapping differs from expected actions",
                    probe.case_id,
                )
            )

    if not adapter_binding_matches() and not any(
        diagnostic.code == "schema.variant_rebound" for diagnostic in diagnostics
    ):
        diagnostics.append(
            _diagnostic(
                "schema.variant_rebound",
                "Adapter variant changed during schema checks",
            )
        )

    return _make_layer_result(
        layer="schema",
        fingerprint=fingerprint,
        checks_run=len(valid_probes),
        diagnostics=diagnostics,
    )


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
            if (
                validated is not snapshot.value
                or current_fingerprint != snapshot.fingerprint
            ):
                raise ValueError("case validation changed entry snapshot")
            if group.capture_episode_ids:
                episode_ids.append(
                    _require_safe_identifier(validated.episode_id, "episode_id")
                )
        except Exception:
            all_valid = False
    frozen_episode_ids = (
        tuple(episode_ids) if group.capture_episode_ids and all_valid else None
    )
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


def evaluate_contract_suite(
    variant: SchemaVariant,
    adapter: SemanticAdapter,
    schema_probes: Sequence[SchemaProbe],
    denotation_cases: Sequence[object],
    state_cases: Sequence[object],
    state_provider: object,
    trace_cases: Sequence[object],
    trace_provider: object,
) -> ContractSuiteResult:
    """Run all four layers and return a sealed, non-short-circuiting suite result."""

    from toolshift.contracts.denotation import (
        DenotationCase,
        _validate_denotation_case,
        check_denotation_contract,
    )
    from toolshift.contracts.state import (
        StateCase,
        _validate_state_case,
        check_state_contract,
    )
    from toolshift.contracts.trace import (
        TraceCase,
        _validate_trace_case,
        check_trace_contract,
    )

    schema_probes = cast(Sequence[SchemaProbe], _freeze_suite_sequence(schema_probes))
    denotation_cases = cast(Sequence[object], _freeze_suite_sequence(denotation_cases))
    state_cases = cast(Sequence[object], _freeze_suite_sequence(state_cases))
    trace_cases = cast(Sequence[object], _freeze_suite_sequence(trace_cases))

    raw_entry_case_groups = (
        _capture_entry_case_group(
            schema_probes,
            SchemaProbe,
            _validate_schema_probe,
        ),
        _capture_entry_case_group(
            denotation_cases,
            DenotationCase,
            _validate_denotation_case,
        ),
        _capture_entry_case_group(
            state_cases,
            StateCase,
            _validate_state_case,
            capture_episode_ids=True,
        ),
        _capture_entry_case_group(
            trace_cases,
            TraceCase,
            _validate_trace_case,
            capture_episode_ids=True,
        ),
    )

    try:
        _validate_schema_variant(variant)
        variant_fingerprint = schema_fingerprint(variant)
        supplied_variant_valid = True
    except Exception:
        supplied_variant_valid = False
        variant_fingerprint = hashlib.sha256(b"toolshift.schema.invalid").hexdigest()

    try:
        adapter_entry_fingerprint = schema_fingerprint(adapter.variant)
    except Exception:
        adapter_entry_fingerprint = None

    entry_case_groups = tuple(
        _validate_entry_case_group(group) for group in raw_entry_case_groups
    )

    def adapter_binding_matches() -> bool:
        try:
            return (
                supplied_variant_valid
                and adapter_entry_fingerprint == variant_fingerprint
                and schema_fingerprint(variant) == variant_fingerprint
                and schema_fingerprint(adapter.variant) == variant_fingerprint
            )
        except Exception:
            return False

    case_snapshot_mismatch_detected = False
    variant_binding_mismatch_detected = False

    def guard_layer_result(result: LayerContractResult) -> LayerContractResult:
        nonlocal case_snapshot_mismatch_detected
        nonlocal variant_binding_mismatch_detected
        extra_diagnostics: list[ContractDiagnostic] = []
        binding_matches = adapter_binding_matches()
        case_snapshots_match = _entry_case_groups_match(entry_case_groups)
        if not binding_matches:
            variant_binding_mismatch_detected = True
        if not case_snapshots_match:
            case_snapshot_mismatch_detected = True
        if case_snapshot_mismatch_detected:
            extra_diagnostics.append(
                _diagnostic(
                    "suite.case_snapshot_mismatch",
                    "Entry case validation or content stability failed",
                )
            )
        if variant_binding_mismatch_detected:
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

    schema_result = guard_layer_result(
        _run_suite_checker(
            "schema",
            variant_fingerprint,
            lambda: check_schema_contract(variant, adapter, schema_probes),
        )
    )
    denotation_result = guard_layer_result(
        _run_suite_checker(
            "denotation",
            hashlib.sha256(b"toolshift.denotation.checker-error").hexdigest(),
            lambda: check_denotation_contract(adapter, denotation_cases),
        )
    )
    state_result = guard_layer_result(
        _run_suite_checker(
            "state",
            hashlib.sha256(b"toolshift.state.checker-error").hexdigest(),
            lambda: check_state_contract(state_cases, state_provider),
        )
    )
    trace_result = guard_layer_result(
        _run_suite_checker(
            "trace",
            hashlib.sha256(b"toolshift.trace.checker-error").hexdigest(),
            lambda: check_trace_contract(
                adapter,
                trace_cases,
                trace_provider,
                denotation_cases,
            ),
        )
    )

    if variant_binding_mismatch_detected and not any(
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

    state_episode_ids = entry_case_groups[2].episode_ids
    if state_episode_ids is None:
        state_episode_fingerprint = hashlib.sha256(
            b"toolshift.state-episodes.invalid"
        ).hexdigest()
    else:
        state_episode_fingerprint = _episode_fingerprint(state_episode_ids)
    trace_episode_ids = entry_case_groups[3].episode_ids
    if trace_episode_ids is None:
        trace_episode_fingerprint = hashlib.sha256(
            b"toolshift.trace-episodes.invalid"
        ).hexdigest()
    else:
        trace_episode_fingerprint = _episode_fingerprint(trace_episode_ids)
    if state_episode_fingerprint != trace_episode_fingerprint:
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

    suite = ContractSuiteResult(
        variant_fingerprint,
        (schema_result, denotation_result, state_result, trace_result),
    )
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
        result_snapshots=tuple(
            result._snapshot_fingerprint for result in suite.results
        ),
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


def _validate_suite_result(
    value: object, *, require_attested: bool = False
) -> ContractSuiteResult:
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
        if type(provenance.results) is not tuple or type(
            provenance.result_snapshots
        ) is not tuple:
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


def require_dataset_admission(
    variant: SchemaVariant,
    suite_result: ContractSuiteResult,
) -> ContractSuiteResult:
    """Reject every suite except a non-vacuous sealed four-layer pass."""

    expected_schema_fingerprint = schema_fingerprint(variant)
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
    "ContractDiagnostic",
    "ContractSuiteResult",
    "LayerContractResult",
    "SchemaProbe",
    "check_schema_contract",
    "contract_suite_fingerprint",
    "evaluate_contract_suite",
    "require_dataset_admission",
    "schema_fingerprint",
]
