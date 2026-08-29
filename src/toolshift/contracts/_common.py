"""Shared immutable results, validation, and attestation primitives."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, cast

from toolshift.types import (
    JSONValue,
    SemanticAction,
    _freeze_mapping,
    _semantic_action_has_canonical_shape,
    canonical_json_bytes,
)

_LAYERS = ("schema", "denotation", "state", "trace")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]{0,127}\Z")
_SAFE_MESSAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .,;:()'=_+-]{0,239}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LAYER_ATTESTATION = object()


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


def _update_framed(digest: object, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _fingerprint_parts(domain: bytes, parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for part in parts:
        _update_framed(digest, part)
    return digest.hexdigest()


def _snapshot_action(value: object, context: str) -> SemanticAction:
    if not _semantic_action_has_canonical_shape(value):
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


def _actions_equal(first: tuple[SemanticAction, ...], second: tuple[SemanticAction, ...]) -> bool:
    return len(first) == len(second) and all(
        left == right for left, right in zip(first, second, strict=True)
    )


@dataclass(frozen=True, slots=True)
class ContractDiagnostic:
    """Payload-free failure metadata safe to persist with contract fingerprints."""

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


def _validate_diagnostic(value: object) -> ContractDiagnostic:
    if type(value) is not ContractDiagnostic:
        raise ValueError("diagnostics must contain only ContractDiagnostic values")
    diagnostic = cast(ContractDiagnostic, value)
    rebuilt = ContractDiagnostic(diagnostic.code, diagnostic.message, diagnostic.case_id)
    if diagnostic != rebuilt:
        raise ValueError("diagnostic has been mutated")
    return diagnostic


@dataclass(frozen=True, slots=True)
class LayerContractResult:
    """Immutable diagnostics and sealed evidence fingerprint for one layer."""

    layer: str
    fingerprint: str
    checks_run: int
    diagnostics: tuple[ContractDiagnostic, ...]
    _snapshot_fingerprint: str = field(init=False, repr=False)
    _attestation: object | None = field(init=False, repr=False, default=None)
    _sealed_snapshot_fingerprint: str | None = field(init=False, repr=False, default=None)
    LAYERS: ClassVar[tuple[str, ...]] = _LAYERS

    def __post_init__(self) -> None:
        if type(self.layer) is not str or self.layer not in self.LAYERS:
            raise ValueError("layer must be schema, denotation, state, or trace")
        object.__setattr__(self, "fingerprint", _require_sha256(self.fingerprint))
        if type(self.checks_run) is not int or self.checks_run < 0:
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
        """Return true exactly when validated diagnostics are empty."""

        result = _validate_layer_result(self)
        return len(result.diagnostics) == 0


def _validate_layer_result(value: object, *, require_attested: bool = False) -> LayerContractResult:
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


def _diagnostic(code: str, message: str, case_id: str = "suite") -> ContractDiagnostic:
    return ContractDiagnostic(code=code, message=message, case_id=case_id)


__all__ = [
    "ContractDiagnostic",
    "LayerContractResult",
]
