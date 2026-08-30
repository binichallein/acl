"""Shared constant-time runtime guards for interface transforms."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.transforms.base import OperatorManifestEntry, TransformValidationError
from toolshift.types import JSONValue, SchemaVariant, SurfaceToolSpec

_ADAPTER_VARIANT_SLOT = SemanticAdapter.__dict__["_variant"]
_ResultT = TypeVar("_ResultT")


def _raw_adapter_variant(adapter: SemanticAdapter) -> object:
    return _ADAPTER_VARIANT_SLOT.__get__(adapter, SemanticAdapter)


def _variant_has_interface_manifest(value: object, error_message: str) -> bool:
    if type(value) is not SchemaVariant:
        return False
    try:
        kind = value.manifest.get("kind")
        if type(kind) is not str:
            return False
        return kind == "toolshift_interface_variant"
    except Exception:
        raise TransformValidationError(error_message) from None


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


def _delegate_with_binding_guard(
    check: Callable[[], None],
    operation: Callable[[], _ResultT],
    expected_error: str,
) -> _ResultT:
    check()
    try:
        try:
            return operation()
        except ValueError:
            raise TransformValidationError(expected_error) from None
    finally:
        check()
