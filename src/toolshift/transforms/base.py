"""Deterministic manifest foundation for behavior-preserving interface transforms."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from toolshift.contracts.schema import schema_fingerprint
from toolshift.types import (
    JSONValue,
    SchemaVariant,
    SurfaceToolSpec,
    _freeze_mapping,
    canonical_json_bytes,
    manifest_sha256,
)

_MAX_SAFE_INTEGER = 2**53 - 1
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OPERATOR_SEED_DOMAIN = b"toolshift.operator-seed.v1\0"
_MANIFEST_SCHEMA_ABI_TAG = b"toolshift.transform-manifest.schema.v1"

TRANSFORM_MANIFEST_SCHEMA_VERSION_HASH = hashlib.sha256(_MANIFEST_SCHEMA_ABI_TAG).hexdigest()


class TransformValidationError(ValueError):
    """A payload-free validation failure at a transform boundary."""


def _require_safe_integer(value: object, context: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_SAFE_INTEGER:
        raise TransformValidationError(
            f"{context} must be an integer in the range [0, {_MAX_SAFE_INTEGER}]"
        )
    return value


def _require_identifier(value: object, context: str) -> str:
    if type(value) is not str or _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise TransformValidationError(f"{context} must be a payload-safe identifier")
    return value


def _require_sha256(value: object, context: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TransformValidationError(f"{context} must be a lowercase SHA256 digest")
    return value


def _validated_base_fingerprint(value: object) -> tuple[SchemaVariant, str]:
    if type(value) is not SchemaVariant:
        raise TransformValidationError("base_variant must be a valid SchemaVariant")
    try:
        fingerprint = schema_fingerprint(value)
    except Exception:
        raise TransformValidationError("base_variant must be a valid SchemaVariant") from None
    return cast(SchemaVariant, value), fingerprint


def _freeze_parameters(value: object) -> Mapping[str, JSONValue]:
    try:
        return _freeze_mapping(value, "parameters")
    except Exception:
        raise TransformValidationError("parameters must be a valid I-JSON mapping") from None


def _entry_snapshot(
    operator_id: str,
    operator: str,
    level: str,
    seed: int,
    version_hash: str,
    parameters: Mapping[str, JSONValue],
) -> bytes:
    return canonical_json_bytes(
        {
            "id": operator_id,
            "operator": operator,
            "level": level,
            "seed": seed,
            "version_hash": version_hash,
            "parameters": parameters,
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class OperatorManifestEntry:
    """One immutable, reproducible operator record in composition order."""

    operator_id: str
    operator: str
    level: str
    seed: int
    version_hash: str
    parameters: Mapping[str, JSONValue]
    _canonical_snapshot: bytes = field(init=False, repr=False)
    _snapshot_fingerprint: str = field(init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        # Freeze the callback-bearing graph before reading scalar fields.  This
        # also makes manual object.__new__ + __init__ construction obey the same
        # attestation ordering as ordinary dataclass construction.
        parameters = _freeze_parameters(self.parameters)
        operator_id = _require_identifier(self.operator_id, "operator_id")
        operator = _require_identifier(self.operator, "operator")
        level = self.level
        if type(level) is not str or level not in {"L1", "L2", "L3"}:
            raise TransformValidationError("level must be one of L1, L2, or L3")
        seed = _require_safe_integer(self.seed, "seed")
        version_hash = _require_sha256(self.version_hash, "version_hash")
        canonical = _entry_snapshot(
            operator_id,
            operator,
            level,
            seed,
            version_hash,
            parameters,
        )
        object.__setattr__(self, "operator_id", operator_id)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "version_hash", version_hash)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "_canonical_snapshot", canonical)
        object.__setattr__(
            self,
            "_snapshot_fingerprint",
            hashlib.sha256(canonical).hexdigest(),
        )

    def as_manifest(self) -> Mapping[str, JSONValue]:
        """Return a newly validated immutable manifest mapping."""

        entry = _validate_operator_entry(self)
        return _freeze_parameters(
            {
                "id": entry.operator_id,
                "operator": entry.operator,
                "level": entry.level,
                "seed": entry.seed,
                "version_hash": entry.version_hash,
                "parameters": entry.parameters,
            }
        )


def _validate_operator_entry(value: object) -> OperatorManifestEntry:
    if type(value) is not OperatorManifestEntry:
        raise TransformValidationError("operators must contain exact OperatorManifestEntry values")
    entry = cast(OperatorManifestEntry, value)
    try:
        # Exhaust any externally supplied Mapping callback before reading the
        # remaining entry fields.  A callback can mutate a frozen dataclass via
        # object.__setattr__; re-reading everything afterwards closes that TOCTOU gap.
        parameters = _freeze_parameters(object.__getattribute__(entry, "parameters"))
        rebuilt = OperatorManifestEntry(
            object.__getattribute__(entry, "operator_id"),
            object.__getattribute__(entry, "operator"),
            object.__getattribute__(entry, "level"),
            object.__getattribute__(entry, "seed"),
            object.__getattribute__(entry, "version_hash"),
            parameters,
        )
        canonical = object.__getattribute__(entry, "_canonical_snapshot")
        snapshot = object.__getattribute__(entry, "_snapshot_fingerprint")
        if (
            type(canonical) is not bytes
            or type(snapshot) is not str
            or canonical != rebuilt._canonical_snapshot
            or snapshot != rebuilt._snapshot_fingerprint
        ):
            raise TransformValidationError("operator entry integrity validation failed")
    except TransformValidationError:
        raise
    except Exception:
        raise TransformValidationError("operator entry integrity validation failed") from None
    return rebuilt


def derive_operator_seed(
    master_seed: int,
    base_schema_fingerprint: str,
    operator_id: str,
    operator: str,
    position: int,
) -> int:
    """Derive one position- and base-bound seed without process-global RNG state."""

    master_seed = _require_safe_integer(master_seed, "master_seed")
    base_schema_fingerprint = _require_sha256(
        base_schema_fingerprint,
        "base_schema_fingerprint",
    )
    operator_id = _require_identifier(operator_id, "operator_id")
    operator = _require_identifier(operator, "operator")
    position = _require_safe_integer(position, "position")
    payload = canonical_json_bytes(
        {
            "master_seed": master_seed,
            "base_schema_fingerprint": base_schema_fingerprint,
            "operator_id": operator_id,
            "operator": operator,
            "position": position,
        }
    )
    digest = hashlib.sha256(_OPERATOR_SEED_DOMAIN + payload).digest()
    return int.from_bytes(digest, "big") % (_MAX_SAFE_INTEGER + 1)


def build_transform_manifest(
    base_variant: SchemaVariant,
    *,
    seed: int,
    operators: tuple[OperatorManifestEntry, ...],
) -> Mapping[str, JSONValue]:
    """Build the canonical non-self-referential transform manifest."""

    if type(operators) is not tuple:
        raise TransformValidationError("operators must be an exact tuple")
    if not operators:
        raise TransformValidationError("operators must be a non-empty tuple")
    seed = _require_safe_integer(seed, "seed")
    # Operator parameter mappings are the only callback-bearing inputs here.
    # Snapshot them before attesting the base so a callback cannot invalidate a
    # fingerprint that the returned manifest would continue to claim.
    validated = tuple(_validate_operator_entry(entry) for entry in operators)
    _, base_fingerprint = _validated_base_fingerprint(base_variant)
    operator_ids = tuple(entry.operator_id for entry in validated)
    if len(operator_ids) != len(set(operator_ids)):
        raise TransformValidationError("operator identifiers must be unique")
    for position, entry in enumerate(validated):
        expected_seed = derive_operator_seed(
            seed,
            base_fingerprint,
            entry.operator_id,
            entry.operator,
            position,
        )
        if entry.seed != expected_seed:
            raise TransformValidationError("operator seed does not match the derived seed")
    return _freeze_parameters(
        {
            "schema_version": 1,
            "kind": "toolshift_interface_variant",
            "base_schema_fingerprint": base_fingerprint,
            "seed": seed,
            "composition_order": operator_ids,
            "version_hash": TRANSFORM_MANIFEST_SCHEMA_VERSION_HASH,
            "operators": tuple(entry.as_manifest() for entry in validated),
        }
    )


def build_transformed_variant(
    base_variant: SchemaVariant,
    tools: tuple[SurfaceToolSpec, ...],
    *,
    seed: int,
    operators: tuple[OperatorManifestEntry, ...],
) -> SchemaVariant:
    """Build a schema variant whose ID is the full canonical manifest digest."""

    _validated_base_fingerprint(base_variant)
    if type(tools) is not tuple or any(type(tool) is not SurfaceToolSpec for tool in tools):
        raise TransformValidationError("tools must be an exact tuple of SurfaceToolSpec values")
    manifest = build_transform_manifest(base_variant, seed=seed, operators=operators)
    variant_id = f"toolshift-v1-{manifest_sha256(manifest)}"
    try:
        variant = SchemaVariant(variant_id, tools, manifest)
        schema_fingerprint(variant)
        return variant
    except Exception:
        raise TransformValidationError("transformed tools do not form a valid schema") from None


__all__ = [
    "TRANSFORM_MANIFEST_SCHEMA_VERSION_HASH",
    "OperatorManifestEntry",
    "TransformValidationError",
    "build_transform_manifest",
    "build_transformed_variant",
    "derive_operator_seed",
]
