"""Schema-layer behavioral-equivalence contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts._common import (
    ContractDiagnostic,
    LayerContractResult,
    _actions_equal,
    _diagnostic,
    _fingerprint_parts,
    _make_layer_result,
    _require_safe_identifier,
    _require_sha256,
    _snapshot_actions,
    _snapshot_call,
    _update_framed,
)
from toolshift.types import (
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    _is_frozen_mapping,
    _schema_variant_has_canonical_shape,
    _semantic_action_has_canonical_shape,
    canonical_json_bytes,
)

_ADAPTER_VARIANT_SLOT = SemanticAdapter.__dict__["_variant"]


def _raw_adapter_variant(adapter: SemanticAdapter) -> object:
    return _ADAPTER_VARIANT_SLOT.__get__(adapter, SemanticAdapter)


def _require_non_blank_text(value: object, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{context} must be a non-blank string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must be valid UTF-8 text") from error
    return value


def _validate_schema_variant(variant: object) -> SchemaVariant:
    if not _schema_variant_has_canonical_shape(variant):
        raise ValueError("variant must be a SchemaVariant")
    typed = cast(SchemaVariant, variant)
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


@dataclass(frozen=True, slots=True, eq=False)
class SchemaProbe:
    """Immutable expected mapping for one declared surface tool call."""

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


def _schema_probe_has_canonical_shape(value: object) -> bool:
    if type(value) is not SchemaProbe:
        return False
    probe = cast(SchemaProbe, value)
    expected_actions = object.__getattribute__(probe, "expected_actions")
    return (
        type(object.__getattribute__(probe, "case_id")) is str
        and type(object.__getattribute__(probe, "surface_tool_name")) is str
        and _is_frozen_mapping(object.__getattribute__(probe, "surface_call"))
        and type(expected_actions) is tuple
        and all(_semantic_action_has_canonical_shape(action) for action in expected_actions)
        and type(object.__getattribute__(probe, "_snapshot_fingerprint")) is str
    )


def _validate_schema_probe(value: object) -> SchemaProbe:
    if not _schema_probe_has_canonical_shape(value):
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


def _call_fingerprint(call: Mapping[str, JSONValue]) -> str:
    return hashlib.sha256(canonical_json_bytes(call)).hexdigest()


def schema_fingerprint(variant: SchemaVariant) -> str:
    """Hash the variant ID, ordered full tool specs, and manifest content."""

    variant = _validate_schema_variant(variant)
    digest = hashlib.sha256(b"toolshift.schema.v1\0")
    _update_framed(digest, canonical_json_bytes(variant.variant_id))
    for tool in variant.tools:
        _update_framed(digest, canonical_json_bytes(tool.name))
        _update_framed(digest, canonical_json_bytes(tool.description))
        _update_framed(digest, canonical_json_bytes(tool.input_schema))
    _update_framed(digest, canonical_json_bytes(variant.manifest))
    return digest.hexdigest()


def _initialize_schema_check(
    variant: SchemaVariant,
    adapter: SemanticAdapter,
    diagnostics: list[ContractDiagnostic],
) -> tuple[str, SchemaVariant | None]:
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
        public_bound_fingerprint = schema_fingerprint(adapter.variant)
        raw_bound_fingerprint = schema_fingerprint(_raw_adapter_variant(adapter))
        if (
            validated_variant is None
            or public_bound_fingerprint != fingerprint
            or raw_bound_fingerprint != fingerprint
        ):
            diagnostics.append(
                _diagnostic(
                    "schema.variant_mismatch",
                    "Adapter variant does not match schema",
                )
            )
    except Exception:
        diagnostics.append(
            _diagnostic("schema.variant_exception", "Adapter variant validation raised")
        )
    return fingerprint, validated_variant


def _schema_binding_matches(
    variant: SchemaVariant,
    adapter: SemanticAdapter,
    validated_variant: SchemaVariant | None,
    fingerprint: str,
) -> bool:
    if validated_variant is None:
        return False
    try:
        return (
            schema_fingerprint(variant) == fingerprint
            and schema_fingerprint(adapter.variant) == fingerprint
            and schema_fingerprint(_raw_adapter_variant(adapter)) == fingerprint
        )
    except Exception:
        return False


def _collect_schema_probes(
    probes: object,
    diagnostics: list[ContractDiagnostic],
) -> list[SchemaProbe]:
    if type(probes) not in (list, tuple):
        diagnostics.append(
            _diagnostic(
                "schema.invalid_probes",
                "Schema probes must be an exact built-in list or tuple",
            )
        )
        probe_values: tuple[object, ...] = ()
    else:
        probe_values = tuple(probes)
    if not probe_values:
        diagnostics.append(_diagnostic("schema.empty_probes", "Schema probes must not be empty"))

    valid_probes: list[SchemaProbe] = []
    for raw_probe in probe_values:
        try:
            valid_probes.append(_validate_schema_probe(raw_probe))
        except Exception:
            try:
                case_id = _require_safe_identifier(
                    getattr(raw_probe, "case_id", "suite"),
                    "case_id",
                )
            except ValueError:
                case_id = "suite"
            diagnostics.append(
                _diagnostic(
                    "schema.invalid_probe",
                    "Schema probe validation failed",
                    case_id,
                )
            )
    return valid_probes


def _check_schema_inventory(
    validated_variant: SchemaVariant | None,
    probes: list[SchemaProbe],
    diagnostics: list[ContractDiagnostic],
) -> None:
    case_ids = [probe.case_id for probe in probes]
    if len(case_ids) != len(set(case_ids)):
        diagnostics.append(
            _diagnostic("schema.duplicate_case", "Schema case identifiers must be unique")
        )
    call_fingerprints = [_call_fingerprint(probe.surface_call) for probe in probes]
    if len(call_fingerprints) != len(set(call_fingerprints)):
        diagnostics.append(
            _diagnostic("schema.duplicate_call", "Schema surface calls must be unique")
        )

    declared_names = (
        {tool.name for tool in validated_variant.tools} if validated_variant is not None else set()
    )
    probed_names = {probe.surface_tool_name for probe in probes}
    if declared_names - probed_names:
        diagnostics.append(
            _diagnostic("schema.missing_tool", "Declared surface tools require probes")
        )
    if probed_names - declared_names:
        diagnostics.append(
            _diagnostic("schema.undeclared_tool", "Probes include undeclared surface tools")
        )


def _check_schema_probe(
    adapter: SemanticAdapter,
    probe: SchemaProbe,
    binding_matches: Callable[[], bool],
    diagnostics: list[ContractDiagnostic],
) -> None:
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
            if not binding_matches():
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
        return
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


def check_schema_contract(
    variant: SchemaVariant,
    adapter: SemanticAdapter,
    probes: list[SchemaProbe] | tuple[SchemaProbe, ...],
) -> LayerContractResult:
    """Verify mappings from an exact built-in list or tuple of schema probes.

    The checker runs every valid probe twice, requires full declared-tool coverage,
    and returns payload-free diagnostics instead of propagating adapter failures.
    """

    diagnostics: list[ContractDiagnostic] = []
    fingerprint, validated_variant = _initialize_schema_check(
        variant,
        adapter,
        diagnostics,
    )

    def adapter_binding_matches() -> bool:
        return _schema_binding_matches(
            variant,
            adapter,
            validated_variant,
            fingerprint,
        )

    valid_probes = _collect_schema_probes(probes, diagnostics)
    _check_schema_inventory(validated_variant, valid_probes, diagnostics)
    for probe in valid_probes:
        _check_schema_probe(adapter, probe, adapter_binding_matches, diagnostics)

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


__all__ = ["SchemaProbe", "check_schema_contract", "schema_fingerprint"]
