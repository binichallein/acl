"""Pure L1 tool-name rename and its behavior-preserving semantic adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar, cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.schema import schema_fingerprint
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
TOOL_NAME_RENAME_VERSION_HASH = hashlib.sha256(_RENAME_ABI_TAG).hexdigest()
_ADAPTER_VARIANT_SLOT = SemanticAdapter.__dict__["_variant"]
_RENAME_INTEGRITY_TOKEN = object()
_ResultT = TypeVar("_ResultT")


def _raw_adapter_variant(adapter: SemanticAdapter) -> object:
    return _ADAPTER_VARIANT_SLOT.__get__(adapter, SemanticAdapter)


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


def _snapshot_full_transform_mapping(value: object) -> Mapping[str, JSONValue]:
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
    return _freeze_transform_mapping(snapshot)


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
) -> Mapping[str, JSONValue]:
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
    return _freeze_transform_mapping(resolved)


def _validated_full_mapping(
    source_variant: SchemaVariant,
    value: object,
) -> Mapping[str, JSONValue]:
    try:
        frozen = _freeze_transform_mapping(value)
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
        return _freeze_transform_mapping(resolved)
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
    mapping: Mapping[str, JSONValue],
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
class _MappingRootSeal:
    mapping: Mapping[str, JSONValue]
    items: object
    index: object


def _make_mapping_root_seal(mapping: Mapping[str, JSONValue]) -> _MappingRootSeal:
    try:
        items = object.__getattribute__(mapping, "_items")
        index = object.__getattribute__(mapping, "_index")
    except Exception:
        raise TransformValidationError("frozen mapping root is invalid") from None
    if type(items) is not tuple or type(index) is not dict:
        raise TransformValidationError("frozen mapping root is invalid")
    return _MappingRootSeal(mapping, items, index)


def _mapping_root_seal_matches(seal: _MappingRootSeal) -> bool:
    try:
        return (
            object.__getattribute__(seal.mapping, "_items") is seal.items
            and object.__getattribute__(seal.mapping, "_index") is seal.index
        )
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class _SchemaRuntimeSeal:
    variant: SchemaVariant
    fingerprint: str
    variant_id: str
    tools: tuple[SurfaceToolSpec, ...]
    manifest: Mapping[str, JSONValue]
    manifest_canonical: bytes
    manifest_root: _MappingRootSeal


def _make_schema_runtime_seal(
    variant: SchemaVariant,
    fingerprint: str,
) -> _SchemaRuntimeSeal:
    tools = object.__getattribute__(variant, "tools")
    manifest = object.__getattribute__(variant, "manifest")
    return _SchemaRuntimeSeal(
        variant=variant,
        fingerprint=fingerprint,
        variant_id=object.__getattribute__(variant, "variant_id"),
        tools=tools,
        manifest=manifest,
        manifest_canonical=object.__getattribute__(variant, "_manifest_canonical"),
        manifest_root=_make_mapping_root_seal(manifest),
    )


def _schema_runtime_seal_matches(seal: _SchemaRuntimeSeal) -> bool:
    try:
        variant = seal.variant
        tools = object.__getattribute__(variant, "tools")
        return not (
            type(variant) is not SchemaVariant
            or object.__getattribute__(variant, "variant_id") != seal.variant_id
            or tools is not seal.tools
            or type(tools) is not tuple
            or object.__getattribute__(variant, "_manifest_canonical") != seal.manifest_canonical
            or object.__getattribute__(variant, "manifest") is not seal.manifest
            or not _mapping_root_seal_matches(seal.manifest_root)
        )
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class _OperatorRuntimeSeal:
    operator: OperatorManifestEntry
    operator_id: str
    operator_name: str
    level: str
    seed: int
    version_hash: str
    parameters: Mapping[str, JSONValue]
    parameters_root: _MappingRootSeal
    canonical_snapshot: bytes
    snapshot_fingerprint: str


def _make_operator_runtime_seal(operator: OperatorManifestEntry) -> _OperatorRuntimeSeal:
    return _OperatorRuntimeSeal(
        operator=operator,
        operator_id=object.__getattribute__(operator, "operator_id"),
        operator_name=object.__getattribute__(operator, "operator"),
        level=object.__getattribute__(operator, "level"),
        seed=object.__getattribute__(operator, "seed"),
        version_hash=object.__getattribute__(operator, "version_hash"),
        parameters=object.__getattribute__(operator, "parameters"),
        parameters_root=_make_mapping_root_seal(object.__getattribute__(operator, "parameters")),
        canonical_snapshot=object.__getattribute__(operator, "_canonical_snapshot"),
        snapshot_fingerprint=object.__getattribute__(operator, "_snapshot_fingerprint"),
    )


def _operator_runtime_seal_matches(seal: _OperatorRuntimeSeal) -> bool:
    try:
        operator = seal.operator
        return (
            type(operator) is OperatorManifestEntry
            and object.__getattribute__(operator, "operator_id") == seal.operator_id
            and object.__getattribute__(operator, "operator") == seal.operator_name
            and object.__getattribute__(operator, "level") == seal.level
            and object.__getattribute__(operator, "seed") == seal.seed
            and object.__getattribute__(operator, "version_hash") == seal.version_hash
            and object.__getattribute__(operator, "parameters") is seal.parameters
            and _mapping_root_seal_matches(seal.parameters_root)
            and object.__getattribute__(operator, "_canonical_snapshot") == seal.canonical_snapshot
            and object.__getattribute__(operator, "_snapshot_fingerprint")
            == seal.snapshot_fingerprint
        )
    except Exception:
        return False


@dataclass(frozen=True, slots=True, eq=False)
class RenameTransform:
    """An immutable pure rename with bidirectional surface-call translation."""

    source_variant: SchemaVariant
    variant: SchemaVariant
    operator: OperatorManifestEntry
    canonical_to_surface: Mapping[str, JSONValue]
    _surface_to_canonical: Mapping[str, JSONValue] = field(init=False, repr=False)
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
        if (
            type(self.source_variant) is not SchemaVariant
            or type(self.variant) is not SchemaVariant
        ):
            raise TransformValidationError("rename variants must be exact SchemaVariant values")
        if type(self.operator) is not OperatorManifestEntry:
            raise TransformValidationError("rename operator must be an OperatorManifestEntry")
        # Exhaust caller-owned mapping callbacks first, including any parameters
        # graph injected into a tampered operator.  Variant attestation happens
        # only after both callback boundaries have completed.
        mapping_snapshot = _snapshot_full_transform_mapping(self.canonical_to_surface)
        try:
            operator_manifest = self.operator.as_manifest()
        except Exception:
            raise TransformValidationError("rename operator integrity validation failed") from None
        try:
            source_fingerprint = schema_fingerprint(self.source_variant)
            variant_fingerprint = schema_fingerprint(self.variant)
        except Exception:
            raise TransformValidationError("rename variant integrity validation failed") from None
        if self.source_variant.manifest.get("kind") == "toolshift_interface_variant":
            raise TransformValidationError(
                "rename composition requires the explicit compose transform"
            )
        mapping = _validated_full_mapping(self.source_variant, mapping_snapshot)
        if (
            self.operator.operator != _RENAME_OPERATOR
            or self.operator.level != _RENAME_LEVEL
            or self.operator.version_hash != TOOL_NAME_RENAME_VERSION_HASH
        ):
            raise TransformValidationError("rename operator metadata is invalid")
        expected_parameters = _freeze_transform_mapping({"mapping": {"tools": mapping}})
        operator_parameters = operator_manifest.get("parameters")
        if not isinstance(operator_parameters, Mapping) or canonical_json_bytes(
            operator_parameters
        ) != canonical_json_bytes(expected_parameters):
            raise TransformValidationError("rename operator parameters do not match mapping")
        try:
            master_seed = self.variant.manifest["seed"]
            if type(master_seed) is not int:
                raise TransformValidationError("rename manifest seed is invalid")
            expected_tools = tuple(
                SurfaceToolSpec(
                    cast(str, mapping[tool.name]),
                    tool.description,
                    tool.input_schema,
                )
                for tool in self.source_variant.tools
            )
            expected_variant = build_transformed_variant(
                self.source_variant,
                expected_tools,
                seed=master_seed,
                operators=(self.operator,),
            )
            if self.variant != expected_variant:
                raise TransformValidationError("rename variant does not match operator manifest")
        except TransformValidationError:
            raise
        except Exception:
            raise TransformValidationError("rename variant integrity validation failed") from None
        inverse = {cast(str, alias): canonical for canonical, alias in mapping.items()}
        inverse_mapping = _freeze_transform_mapping(inverse)
        canonical = _transform_snapshot(
            source_fingerprint,
            variant_fingerprint,
            operator_manifest,
            mapping,
        )
        object.__setattr__(self, "canonical_to_surface", mapping)
        object.__setattr__(self, "_surface_to_canonical", inverse_mapping)
        object.__setattr__(self, "_source_schema_fingerprint", source_fingerprint)
        object.__setattr__(self, "_variant_schema_fingerprint", variant_fingerprint)
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
            _make_schema_runtime_seal(self.source_variant, source_fingerprint),
        )
        object.__setattr__(
            self,
            "_variant_runtime_seal",
            _make_schema_runtime_seal(self.variant, variant_fingerprint),
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
        mapping = cast(Mapping[str, str], self._surface_to_canonical)
        return _replace_call_name(call, mapping, context="surface call")

    def canonical_call_to_surface(
        self,
        call: Mapping[str, JSONValue],
    ) -> Mapping[str, JSONValue]:
        """Translate only a canonical source call into the final interface."""

        self._require_integrity()
        mapping = cast(Mapping[str, str], self.canonical_to_surface)
        return _replace_call_name(call, mapping, context="canonical call")

    def _trace_for_source(self, trace: ExecutionTrace) -> ExecutionTrace:
        self._require_integrity()
        try:
            if type(trace) is not ExecutionTrace or not _execution_trace_has_canonical_shape(trace):
                raise TransformValidationError("trace is invalid")
            rebuilt = ExecutionTrace(trace.surface_calls, trace.semantic_actions, trace.base_calls)
            if rebuilt != trace:
                raise TransformValidationError("trace is invalid")
            canonical_names = set(self.canonical_to_surface)
            changed_canonical = {
                canonical
                for canonical, alias in self.canonical_to_surface.items()
                if canonical != alias
            }
            changed_surface = {
                cast(str, alias)
                for canonical, alias in self.canonical_to_surface.items()
                if canonical != alias
            }
            saw_canonical = False
            saw_surface = False
            for call in trace.surface_calls:
                name = call.get("name")
                if type(name) is not str:
                    raise TransformValidationError("trace surface call is invalid")
                if name in changed_canonical:
                    saw_canonical = True
                elif name in changed_surface:
                    saw_surface = True
                elif name not in canonical_names:
                    raise TransformValidationError("trace surface call has an unknown tool")
            if saw_canonical and saw_surface:
                raise TransformValidationError("trace mixes canonical and renamed surface modes")
            if not saw_surface:
                return trace
            inverse = cast(Mapping[str, str], self._surface_to_canonical)
            translated = tuple(
                _replace_call_name(call, inverse, context="trace surface call")
                for call in trace.surface_calls
            )
            return ExecutionTrace(translated, trace.semantic_actions, trace.base_calls)
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
        raise TransformValidationError("rename composition requires the explicit compose transform")
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
            cast(str, resolved[tool.name]),
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
            raise TransformValidationError(
                "rename composition requires the explicit compose transform"
            )
        if type(transform) is not RenameTransform:
            raise TransformValidationError("transform must be a RenameTransform")
        transform._require_integrity()
        source_variant = transform.source_variant
        if source_variant.manifest.get("kind") == "toolshift_interface_variant":
            raise TransformValidationError(
                "rename composition requires the explicit compose transform"
            )
        try:
            source_public = source_adapter.variant
            source_raw = _raw_adapter_variant(source_adapter)
        except Exception:
            raise TransformValidationError("source adapter binding validation failed") from None
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
        self._require_bindings()
        try:
            try:
                return operation()
            except ValueError:
                raise TransformValidationError(expected_error) from None
        finally:
            self._require_bindings()

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
        raise TransformValidationError("rename composition requires the explicit compose transform")
    try:
        source_variant = _raw_adapter_variant(source_adapter)
    except Exception:
        raise TransformValidationError("source adapter binding validation failed") from None
    if type(source_variant) is not SchemaVariant:
        raise TransformValidationError("source adapter binding validation failed")
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
