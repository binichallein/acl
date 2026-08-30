"""Pure L1 tool-name rename and its behavior-preserving semantic adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.schema import schema_fingerprint
from toolshift.transforms._runtime import (
    _delegate_with_binding_guard,
    _make_mapping_root_seal,
    _make_operator_runtime_seal,
    _make_schema_runtime_seal,
    _mapping_root_seal_matches,
    _MappingRootSeal,
    _operator_runtime_seal_matches,
    _OperatorRuntimeSeal,
    _raw_adapter_variant,
    _schema_runtime_seal_matches,
    _SchemaRuntimeSeal,
    _variant_has_interface_manifest,
)
from toolshift.transforms.base import (
    OperatorManifestEntry,
    TransformValidationError,
    build_transformed_variant,
    derive_operator_seed,
)
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    _execution_trace_has_canonical_shape,
    _freeze_mapping,
    canonical_json_bytes,
)

_RENAME_OPERATOR = "tool_name_rename"
_RENAME_LEVEL = "L1"
_RENAME_ABI_TAG = b"toolshift.transform.tool-name-rename.v1"
_COMPOSITION_UNSUPPORTED_MESSAGE = "rename composition requires the explicit compose transform"
TOOL_NAME_RENAME_VERSION_HASH = hashlib.sha256(_RENAME_ABI_TAG).hexdigest()
_RENAME_INTEGRITY_TOKEN = object()
_ResultT = TypeVar("_ResultT")


def _require_tool_name(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise TransformValidationError("tool aliases must be non-blank strings")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise TransformValidationError("tool aliases must be valid UTF-8 text") from None
    return value


def _freeze_transform_mapping(value: object) -> Mapping[str, JSONValue]:
    try:
        return _freeze_mapping(value, "transform_mapping")
    except Exception:
        raise TransformValidationError("transform mapping must be valid I-JSON") from None


def _freeze_name_mapping(value: object) -> Mapping[str, str]:
    frozen = _freeze_transform_mapping(value)
    if any(type(alias) is not str for alias in frozen.values()):
        raise TransformValidationError("name mapping values must be strings")
    return cast(Mapping[str, str], frozen)


def _snapshot_full_transform_mapping(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TransformValidationError("transform mapping must be valid I-JSON")
    try:
        raw_entries = tuple(value.items())
        entries = tuple(tuple(entry) for entry in raw_entries)
    except Exception:
        raise TransformValidationError("transform mapping must be readable") from None
    snapshot: dict[str, object] = {}
    for entry in entries:
        if len(entry) != 2 or type(entry[0]) is not str:
            raise TransformValidationError("transform mapping must be valid I-JSON")
        key = entry[0]
        if key in snapshot:
            raise TransformValidationError("transform mapping contains a duplicate canonical tool")
        snapshot[key] = entry[1]
    return _freeze_name_mapping(snapshot)


def _snapshot_external_mapping_entries(
    supplied: object,
) -> tuple[tuple[object, object], ...]:
    if not isinstance(supplied, Mapping):
        raise TransformValidationError("tool_name_mapping must be a mapping")
    try:
        raw_entries = tuple(supplied.items())
        entries = tuple(tuple(entry) for entry in raw_entries)
    except Exception:
        raise TransformValidationError("tool_name_mapping must be a readable mapping") from None
    if any(len(entry) != 2 for entry in entries):
        raise TransformValidationError("tool_name_mapping must contain key-value entries")
    return tuple((entry[0], entry[1]) for entry in entries)


def _resolved_external_mapping(
    base_variant: SchemaVariant,
    entries: tuple[tuple[object, object], ...],
) -> Mapping[str, str]:
    canonical_names = tuple(tool.name for tool in base_variant.tools)
    canonical_set = set(canonical_names)
    if not entries:
        raise TransformValidationError("tool_name_mapping must contain a real rename")
    explicit: dict[str, str] = {}
    for key, raw_alias in entries:
        if type(key) is not str or key not in canonical_set:
            raise TransformValidationError("tool_name_mapping contains an unknown canonical tool")
        if key in explicit:
            raise TransformValidationError("tool_name_mapping contains a duplicate canonical tool")
        alias = _require_tool_name(raw_alias)
        if alias == key:
            raise TransformValidationError("explicit identity mappings are not allowed")
        explicit[key] = alias
    resolved = {name: explicit.get(name, name) for name in canonical_names}
    aliases = tuple(resolved.values())
    if len(aliases) != len(set(aliases)):
        raise TransformValidationError("resolved tool aliases must be unique")
    for canonical, alias in resolved.items():
        if alias != canonical and alias in canonical_set:
            raise TransformValidationError("renamed aliases must be fresh canonical names")
    return _freeze_name_mapping(resolved)


def _validated_full_mapping(
    source_variant: SchemaVariant,
    value: object,
) -> Mapping[str, str]:
    try:
        frozen = _freeze_name_mapping(value)
        canonical_names = tuple(tool.name for tool in source_variant.tools)
        if set(frozen) != set(canonical_names):
            raise TransformValidationError("rename mapping does not cover every canonical tool")
        resolved: dict[str, str] = {}
        for name in canonical_names:
            alias = _require_tool_name(frozen[name])
            resolved[name] = alias
        aliases = tuple(resolved.values())
        if len(aliases) != len(set(aliases)):
            raise TransformValidationError("resolved tool aliases must be unique")
        if not any(name != alias for name, alias in resolved.items()):
            raise TransformValidationError("rename mapping must contain a real rename")
        canonical_set = set(canonical_names)
        if any(alias != name and alias in canonical_set for name, alias in resolved.items()):
            raise TransformValidationError("renamed aliases must be fresh canonical names")
        return _freeze_name_mapping(resolved)
    except TransformValidationError:
        raise
    except Exception:
        raise TransformValidationError("rename mapping integrity validation failed") from None


def _replace_call_name(
    call: object,
    mapping: Mapping[str, str],
    *,
    context: str,
) -> Mapping[str, JSONValue]:
    try:
        frozen_call = _freeze_mapping(call, context)
    except Exception:
        raise TransformValidationError(f"{context} is invalid") from None
    name = frozen_call.get("name")
    if type(name) is not str or name not in mapping:
        raise TransformValidationError(f"{context} has an unknown tool name")
    try:
        replaced = {
            key: mapping[name] if key == "name" else value for key, value in frozen_call.items()
        }
        return _freeze_mapping(replaced, context)
    except Exception:
        raise TransformValidationError(f"{context} is invalid") from None


def _transform_snapshot(
    source_fingerprint: str,
    variant_fingerprint: str,
    operator_manifest: Mapping[str, JSONValue],
    mapping: Mapping[str, str],
) -> bytes:
    return canonical_json_bytes(
        {
            "source_schema_fingerprint": source_fingerprint,
            "variant_schema_fingerprint": variant_fingerprint,
            "operator": operator_manifest,
            "canonical_to_surface": mapping,
        }
    )


@dataclass(frozen=True, slots=True)
class _ValidatedRenameConstruction:
    mapping: Mapping[str, str]
    operator_manifest: Mapping[str, JSONValue]
    source_fingerprint: str
    variant_fingerprint: str


def _attest_rename_callback_boundaries(
    source_variant: object,
    variant: object,
    operator: object,
    canonical_to_surface: object,
) -> tuple[
    SchemaVariant,
    SchemaVariant,
    OperatorManifestEntry,
    Mapping[str, str],
    Mapping[str, JSONValue],
    str,
    str,
]:
    if type(source_variant) is not SchemaVariant or type(variant) is not SchemaVariant:
        raise TransformValidationError("rename variants must be exact SchemaVariant values")
    if type(operator) is not OperatorManifestEntry:
        raise TransformValidationError("rename operator must be an OperatorManifestEntry")
    typed_source = cast(SchemaVariant, source_variant)
    typed_variant = cast(SchemaVariant, variant)
    typed_operator = cast(OperatorManifestEntry, operator)

    # Exhaust caller-owned mapping callbacks first, including any parameters
    # graph injected into a tampered operator. Variant attestation happens only
    # after both callback boundaries have completed.
    mapping_snapshot = _snapshot_full_transform_mapping(canonical_to_surface)
    try:
        operator_manifest = typed_operator.as_manifest()
    except Exception:
        raise TransformValidationError("rename operator integrity validation failed") from None
    try:
        source_fingerprint = schema_fingerprint(typed_source)
        variant_fingerprint = schema_fingerprint(typed_variant)
    except Exception:
        raise TransformValidationError("rename variant integrity validation failed") from None
    if typed_source.manifest.get("kind") == "toolshift_interface_variant":
        raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
    return (
        typed_source,
        typed_variant,
        typed_operator,
        mapping_snapshot,
        operator_manifest,
        source_fingerprint,
        variant_fingerprint,
    )


def _require_rename_operator_matches_mapping(
    operator: OperatorManifestEntry,
    operator_manifest: Mapping[str, JSONValue],
    mapping: Mapping[str, str],
) -> None:
    if (
        operator.operator != _RENAME_OPERATOR
        or operator.level != _RENAME_LEVEL
        or operator.version_hash != TOOL_NAME_RENAME_VERSION_HASH
    ):
        raise TransformValidationError("rename operator metadata is invalid")
    expected_parameters = _freeze_transform_mapping({"mapping": {"tools": mapping}})
    operator_parameters = operator_manifest.get("parameters")
    if not isinstance(operator_parameters, Mapping) or canonical_json_bytes(
        operator_parameters
    ) != canonical_json_bytes(expected_parameters):
        raise TransformValidationError("rename operator parameters do not match mapping")


def _require_rename_variant_matches(
    source_variant: SchemaVariant,
    variant: SchemaVariant,
    operator: OperatorManifestEntry,
    mapping: Mapping[str, str],
) -> None:
    try:
        master_seed = variant.manifest["seed"]
        if type(master_seed) is not int:
            raise TransformValidationError("rename manifest seed is invalid")
        expected_tools = tuple(
            SurfaceToolSpec(
                mapping[tool.name],
                tool.description,
                tool.input_schema,
            )
            for tool in source_variant.tools
        )
        expected_variant = build_transformed_variant(
            source_variant,
            expected_tools,
            seed=master_seed,
            operators=(operator,),
        )
        if variant != expected_variant:
            raise TransformValidationError("rename variant does not match operator manifest")
    except TransformValidationError:
        raise
    except Exception:
        raise TransformValidationError("rename variant integrity validation failed") from None


def _validate_rename_construction(
    source_variant: object,
    variant: object,
    operator: object,
    canonical_to_surface: object,
) -> _ValidatedRenameConstruction:
    (
        typed_source,
        typed_variant,
        typed_operator,
        mapping_snapshot,
        operator_manifest,
        source_fingerprint,
        variant_fingerprint,
    ) = _attest_rename_callback_boundaries(
        source_variant,
        variant,
        operator,
        canonical_to_surface,
    )
    mapping = _validated_full_mapping(typed_source, mapping_snapshot)
    _require_rename_operator_matches_mapping(
        typed_operator,
        operator_manifest,
        mapping,
    )
    _require_rename_variant_matches(
        typed_source,
        typed_variant,
        typed_operator,
        mapping,
    )
    return _ValidatedRenameConstruction(
        mapping,
        operator_manifest,
        source_fingerprint,
        variant_fingerprint,
    )


def _validated_trace_snapshot(trace: object) -> ExecutionTrace:
    if type(trace) is not ExecutionTrace or not _execution_trace_has_canonical_shape(trace):
        raise TransformValidationError("trace is invalid")
    typed_trace = cast(ExecutionTrace, trace)
    rebuilt = ExecutionTrace(
        typed_trace.surface_calls,
        typed_trace.semantic_actions,
        typed_trace.base_calls,
    )
    if rebuilt != typed_trace:
        raise TransformValidationError("trace is invalid")
    return typed_trace


def _trace_name_uses_final_surface(
    name: object,
    canonical_names: set[str],
    changed_canonical: set[str],
    changed_surface: set[str],
) -> bool | None:
    if type(name) is not str:
        raise TransformValidationError("trace surface call is invalid")
    if name in changed_canonical:
        return False
    if name in changed_surface:
        return True
    if name not in canonical_names:
        raise TransformValidationError("trace surface call has an unknown tool")
    return None


def _trace_uses_final_surface(
    trace: ExecutionTrace,
    mapping: Mapping[str, str],
) -> bool:
    canonical_names = set(mapping)
    changed_canonical = {canonical for canonical, alias in mapping.items() if canonical != alias}
    changed_surface = {alias for canonical, alias in mapping.items() if canonical != alias}
    modes = {
        _trace_name_uses_final_surface(
            call.get("name"),
            canonical_names,
            changed_canonical,
            changed_surface,
        )
        for call in trace.surface_calls
    }
    if False in modes and True in modes:
        raise TransformValidationError("trace mixes canonical and renamed surface modes")
    return True in modes


def _translate_trace_to_source(
    trace: ExecutionTrace,
    inverse_mapping: Mapping[str, str],
) -> ExecutionTrace:
    translated = tuple(
        _replace_call_name(
            call,
            inverse_mapping,
            context="trace surface call",
        )
        for call in trace.surface_calls
    )
    return ExecutionTrace(translated, trace.semantic_actions, trace.base_calls)


@dataclass(frozen=True, slots=True, eq=False)
class RenameTransform:
    """An immutable pure rename with bidirectional surface-call translation."""

    source_variant: SchemaVariant
    variant: SchemaVariant
    operator: OperatorManifestEntry
    canonical_to_surface: Mapping[str, str]
    _surface_to_canonical: Mapping[str, str] = field(init=False, repr=False)
    _source_schema_fingerprint: str = field(init=False, repr=False)
    _variant_schema_fingerprint: str = field(init=False, repr=False)
    _canonical_snapshot: bytes = field(init=False, repr=False)
    _snapshot_fingerprint: str = field(init=False, repr=False)
    _source_runtime_seal: _SchemaRuntimeSeal = field(init=False, repr=False)
    _variant_runtime_seal: _SchemaRuntimeSeal = field(init=False, repr=False)
    _operator_runtime_seal: _OperatorRuntimeSeal = field(init=False, repr=False)
    _canonical_mapping_root: _MappingRootSeal = field(init=False, repr=False)
    _inverse_mapping_root: _MappingRootSeal = field(init=False, repr=False)
    _runtime_canonical_snapshot: bytes = field(init=False, repr=False)
    _runtime_snapshot_fingerprint: str = field(init=False, repr=False)
    _integrity_token: object = field(init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        construction = _validate_rename_construction(
            self.source_variant,
            self.variant,
            self.operator,
            self.canonical_to_surface,
        )
        self._install_validated_construction(construction)

    def _install_validated_construction(
        self,
        construction: _ValidatedRenameConstruction,
    ) -> None:
        mapping = construction.mapping
        inverse = {alias: canonical for canonical, alias in mapping.items()}
        inverse_mapping = _freeze_name_mapping(inverse)
        canonical = _transform_snapshot(
            construction.source_fingerprint,
            construction.variant_fingerprint,
            construction.operator_manifest,
            mapping,
        )
        object.__setattr__(self, "canonical_to_surface", mapping)
        object.__setattr__(self, "_surface_to_canonical", inverse_mapping)
        object.__setattr__(
            self,
            "_source_schema_fingerprint",
            construction.source_fingerprint,
        )
        object.__setattr__(
            self,
            "_variant_schema_fingerprint",
            construction.variant_fingerprint,
        )
        object.__setattr__(self, "_canonical_snapshot", canonical)
        snapshot_fingerprint = hashlib.sha256(canonical).hexdigest()
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            snapshot_fingerprint,
        )
        object.__setattr__(
            self,
            "_source_runtime_seal",
            _make_schema_runtime_seal(
                self.source_variant,
                construction.source_fingerprint,
            ),
        )
        object.__setattr__(
            self,
            "_variant_runtime_seal",
            _make_schema_runtime_seal(
                self.variant,
                construction.variant_fingerprint,
            ),
        )
        object.__setattr__(
            self,
            "_operator_runtime_seal",
            _make_operator_runtime_seal(self.operator),
        )
        object.__setattr__(
            self,
            "_canonical_mapping_root",
            _make_mapping_root_seal(mapping),
        )
        object.__setattr__(
            self,
            "_inverse_mapping_root",
            _make_mapping_root_seal(inverse_mapping),
        )
        object.__setattr__(self, "_runtime_canonical_snapshot", canonical)
        object.__setattr__(
            self,
            "_runtime_snapshot_fingerprint",
            snapshot_fingerprint,
        )
        object.__setattr__(self, "_integrity_token", _RENAME_INTEGRITY_TOKEN)

    def _require_integrity(self) -> None:
        # Construction and Gate admission deep-validate the frozen JSON graph.
        # Online rollout guards intentionally attest only roots in O(1): code
        # able to rewrite private nested nodes with object.__setattr__ could also
        # rewrite any in-process deep seal, while an O(schema) scan per call would
        # make realistic agentic-RL rollouts impractical.
        try:
            if (
                object.__getattribute__(self, "_integrity_token") is not _RENAME_INTEGRITY_TOKEN
                or object.__getattribute__(self, "source_variant")
                is not self._source_runtime_seal.variant
                or object.__getattribute__(self, "variant")
                is not self._variant_runtime_seal.variant
                or object.__getattribute__(self, "operator")
                is not self._operator_runtime_seal.operator
                or object.__getattribute__(self, "_source_schema_fingerprint")
                != self._source_runtime_seal.fingerprint
                or object.__getattribute__(self, "_variant_schema_fingerprint")
                != self._variant_runtime_seal.fingerprint
                or object.__getattribute__(self, "_canonical_snapshot")
                != self._runtime_canonical_snapshot
                or object.__getattribute__(self, "_snapshot_fingerprint")
                != self._runtime_snapshot_fingerprint
                or not _schema_runtime_seal_matches(self._source_runtime_seal)
                or not _schema_runtime_seal_matches(self._variant_runtime_seal)
                or not _operator_runtime_seal_matches(self._operator_runtime_seal)
                or object.__getattribute__(self, "canonical_to_surface")
                is not self._canonical_mapping_root.mapping
                or not _mapping_root_seal_matches(self._canonical_mapping_root)
                or object.__getattribute__(self, "_surface_to_canonical")
                is not self._inverse_mapping_root.mapping
                or not _mapping_root_seal_matches(self._inverse_mapping_root)
            ):
                raise TransformValidationError("rename transform integrity validation failed")
        except Exception:
            raise TransformValidationError("rename transform integrity validation failed") from None

    def surface_call_to_canonical(
        self,
        call: Mapping[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        """Translate only a final-surface call into the source interface."""

        self._require_integrity()
        return _replace_call_name(
            call,
            self._surface_to_canonical,
            context="surface call",
        )

    def canonical_call_to_surface(
        self,
        call: Mapping[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        """Translate only a canonical source call into the final interface."""

        self._require_integrity()
        return _replace_call_name(
            call,
            self.canonical_to_surface,
            context="canonical call",
        )

    def _trace_for_source(self, trace: ExecutionTrace) -> ExecutionTrace:
        self._require_integrity()
        try:
            validated = _validated_trace_snapshot(trace)
            if not _trace_uses_final_surface(validated, self.canonical_to_surface):
                return validated
            return _translate_trace_to_source(validated, self._surface_to_canonical)
        except TransformValidationError:
            raise
        except Exception:
            raise TransformValidationError("trace is invalid") from None


def build_rename_transform(
    base_variant: SchemaVariant,
    *,
    tool_name_mapping: Mapping[str, str],
    seed: int,
    operator_id: str = "rename-000",
) -> RenameTransform:
    """Build one pure partial rename expanded to a full deterministic mapping."""

    if type(base_variant) is not SchemaVariant:
        raise TransformValidationError("base_variant must be a valid SchemaVariant")
    entries = _snapshot_external_mapping_entries(tool_name_mapping)
    try:
        base_fingerprint = schema_fingerprint(base_variant)
    except Exception:
        raise TransformValidationError("base_variant must be a valid SchemaVariant") from None
    if base_variant.manifest.get("kind") == "toolshift_interface_variant":
        raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
    resolved = _resolved_external_mapping(base_variant, entries)
    operator_seed = derive_operator_seed(
        seed,
        base_fingerprint,
        operator_id,
        _RENAME_OPERATOR,
        0,
    )
    operator = OperatorManifestEntry(
        operator_id,
        _RENAME_OPERATOR,
        _RENAME_LEVEL,
        operator_seed,
        TOOL_NAME_RENAME_VERSION_HASH,
        {"mapping": {"tools": resolved}},
    )
    tools = tuple(
        SurfaceToolSpec(
            resolved[tool.name],
            tool.description,
            tool.input_schema,
        )
        for tool in base_variant.tools
    )
    variant = build_transformed_variant(
        base_variant,
        tools,
        seed=seed,
        operators=(operator,),
    )
    return RenameTransform(base_variant, variant, operator, resolved)


class RenameAdapter(SemanticAdapter):
    """Delegate a pure renamed interface to an exact source semantic adapter."""

    __slots__ = (
        "_final_schema_fingerprint",
        "_final_variant",
        "_source_adapter",
        "_source_adapter_seal",
        "_source_schema_fingerprint",
        "_source_variant",
        "_transform",
        "_transform_seal",
    )

    def __init__(self, source_adapter: SemanticAdapter, transform: RenameTransform) -> None:
        if not isinstance(source_adapter, SemanticAdapter):
            raise TransformValidationError("source_adapter must be a SemanticAdapter")
        if isinstance(source_adapter, RenameAdapter):
            raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
        if type(transform) is not RenameTransform:
            raise TransformValidationError("transform must be a RenameTransform")
        transform._require_integrity()
        source_variant = transform.source_variant
        if source_variant.manifest.get("kind") == "toolshift_interface_variant":
            raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
        try:
            source_public = source_adapter.variant
            source_raw = _raw_adapter_variant(source_adapter)
        except Exception:
            raise TransformValidationError("source adapter binding validation failed") from None
        if _variant_has_interface_manifest(
            source_public,
            "source adapter binding validation failed",
        ) or _variant_has_interface_manifest(
            source_raw,
            "source adapter binding validation failed",
        ):
            raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
        if source_public is not source_variant or source_raw is not source_variant:
            raise TransformValidationError("source adapter binding does not match transform")
        try:
            source_fingerprint = schema_fingerprint(source_variant)
            public_fingerprint = schema_fingerprint(source_public)
            raw_fingerprint = schema_fingerprint(source_raw)
            final_fingerprint = schema_fingerprint(transform.variant)
        except Exception:
            raise TransformValidationError("source adapter binding validation failed") from None
        if public_fingerprint != source_fingerprint or raw_fingerprint != source_fingerprint:
            raise TransformValidationError("source adapter binding does not match transform")
        super().__init__(transform.variant)
        self._source_adapter = source_adapter
        self._source_adapter_seal = source_adapter
        self._transform = transform
        self._transform_seal = transform
        self._source_variant = source_variant
        self._source_schema_fingerprint = source_fingerprint
        self._final_variant = transform.variant
        self._final_schema_fingerprint = final_fingerprint
        self._require_bindings()

    @property
    def transform(self) -> RenameTransform:
        """The immutable rename specification handled by this adapter."""

        return self._transform

    def _require_bindings(self) -> None:
        try:
            if (
                self._transform is not self._transform_seal
                or self._source_adapter is not self._source_adapter_seal
                or self._transform.source_variant is not self._source_variant
                or self._transform.variant is not self._final_variant
                or self._source_schema_fingerprint
                != self._transform._source_runtime_seal.fingerprint
                or self._final_schema_fingerprint
                != self._transform._variant_runtime_seal.fingerprint
            ):
                raise TransformValidationError("adapter binding integrity validation failed")
            self._transform._require_integrity()
        except Exception:
            raise TransformValidationError("adapter binding integrity validation failed") from None
        try:
            source_public = self._source_adapter.variant
            source_raw = _raw_adapter_variant(self._source_adapter)
            final_public = self.variant
            final_raw = _raw_adapter_variant(self)
        except Exception:
            raise TransformValidationError("adapter binding integrity validation failed") from None
        if (
            source_public is not self._source_variant
            or source_raw is not self._source_variant
            or final_public is not self._final_variant
            or final_raw is not self._final_variant
        ):
            raise TransformValidationError("adapter binding integrity validation failed")

    def _delegate(
        self,
        operation: Callable[[], _ResultT],
        expected_error: str,
    ) -> _ResultT:
        return _delegate_with_binding_guard(
            self._require_bindings,
            operation,
            expected_error,
        )

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        canonical_call = self._transform.surface_call_to_canonical(surface_call)
        return self._delegate(
            lambda: self._source_adapter.surface_to_semantic(canonical_call),
            "source adapter rejected the translated surface call",
        )

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        return self._delegate(
            lambda: self._source_adapter.semantic_to_base_calls(action),
            "source adapter rejected the semantic action",
        )

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        canonical_call = self._transform.surface_call_to_canonical(surface_call)
        return self._delegate(
            lambda: self._source_adapter.base_observation_to_surface(
                canonical_call,
                actions,
                base_observation_groups,
            ),
            "source adapter rejected observation wrapping",
        )

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        canonical_trace = self._transform._trace_for_source(trace)
        return self._delegate(
            lambda: self._source_adapter.canonicalize_trace(canonical_trace),
            "source adapter rejected trace canonicalization",
        )


def apply_rename(
    source_adapter: SemanticAdapter,
    *,
    tool_name_mapping: Mapping[str, str],
    seed: int,
    operator_id: str = "rename-000",
) -> RenameAdapter:
    """Build and bind a pure rename to the source adapter's exact raw variant."""

    if not isinstance(source_adapter, SemanticAdapter):
        raise TransformValidationError("source_adapter must be a SemanticAdapter")
    if isinstance(source_adapter, RenameAdapter):
        raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
    try:
        source_variant = _raw_adapter_variant(source_adapter)
    except Exception:
        raise TransformValidationError("source adapter binding validation failed") from None
    if type(source_variant) is not SchemaVariant:
        raise TransformValidationError("source adapter binding validation failed")
    if _variant_has_interface_manifest(
        source_variant,
        "source adapter binding validation failed",
    ):
        raise TransformValidationError(_COMPOSITION_UNSUPPORTED_MESSAGE)
    transform = build_rename_transform(
        source_variant,
        tool_name_mapping=tool_name_mapping,
        seed=seed,
        operator_id=operator_id,
    )
    return RenameAdapter(source_adapter, transform)


__all__ = [
    "TOOL_NAME_RENAME_VERSION_HASH",
    "RenameAdapter",
    "RenameTransform",
    "apply_rename",
    "build_rename_transform",
]
