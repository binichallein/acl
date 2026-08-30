"""Package-internal contracts for shared transform runtime guards."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from toolshift.contracts.schema import schema_fingerprint
from toolshift.transforms._runtime import (
    _delegate_with_binding_guard,
    _make_mapping_root_seal,
    _make_operator_runtime_seal,
    _make_schema_runtime_seal,
    _mapping_root_seal_matches,
    _operator_runtime_seal_matches,
    _schema_runtime_seal_matches,
    _variant_has_interface_manifest,
)
from toolshift.transforms.base import OperatorManifestEntry, TransformValidationError
from toolshift.types import SchemaVariant, SurfaceToolSpec


def _variant() -> SchemaVariant:
    return SchemaVariant(
        "runtime-v1",
        (
            SurfaceToolSpec(
                "search",
                "Search records",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "options": {
                            "type": "object",
                            "properties": {"limit": {"type": "integer"}},
                        },
                    },
                },
            ),
        ),
        {"operator": "identity", "metadata": {"nested": True}},
    )


def _operator() -> OperatorManifestEntry:
    return OperatorManifestEntry(
        "runtime-000",
        "parameter_restructure",
        "L2",
        7,
        "a" * 64,
        {"rules": {"search": {"container": "request"}}},
    )


class _TruthinessBomb:
    def __bool__(self) -> bool:
        raise RuntimeError("PRIVATE_TRUTH_PAYLOAD")


class _EqualityBomb:
    def __eq__(self, other: object) -> object:
        return _TruthinessBomb()


class _EqualityBombManifest(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        return _EqualityBomb()

    def __iter__(self):
        return iter(("kind",))

    def __len__(self) -> int:
        return 1


def test_interface_manifest_probe_returns_exact_bool_without_external_equality() -> None:
    variant = _variant()
    object.__setattr__(variant, "manifest", _EqualityBombManifest())

    result = _variant_has_interface_manifest(variant, "manifest probe failed")

    assert result is False


def test_delegate_checks_before_and_after_operation_and_preserves_identity() -> None:
    events: list[str] = []
    sentinel = object()

    result = _delegate_with_binding_guard(
        lambda: events.append("check"),
        lambda: events.append("operation") or sentinel,
        "source rejected operation",
    )

    assert result is sentinel
    assert events == ["check", "operation", "check"]


def test_delegate_converts_source_value_error_to_static_payload_free_error() -> None:
    events: list[str] = []

    def operation() -> object:
        events.append("operation")
        raise ValueError("PRIVATE")

    with pytest.raises(TransformValidationError) as caught:
        _delegate_with_binding_guard(
            lambda: events.append("check"),
            operation,
            "source rejected operation",
        )

    assert str(caught.value) == "source rejected operation"
    assert "PRIVATE" not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert events == ["check", "operation", "check"]


def test_delegate_propagates_unexpected_error_when_final_check_succeeds() -> None:
    events: list[str] = []
    source_error = RuntimeError("unexpected")

    def operation() -> object:
        events.append("operation")
        raise source_error

    with pytest.raises(RuntimeError) as caught:
        _delegate_with_binding_guard(
            lambda: events.append("check"),
            operation,
            "source rejected operation",
        )

    assert caught.value is source_error
    assert events == ["check", "operation", "check"]


@pytest.mark.parametrize("outcome", ["return", "value_error", "runtime_error"])
def test_delegate_final_binding_failure_overrides_callback_outcome(outcome: str) -> None:
    events: list[str] = []
    final_error = TransformValidationError("adapter binding integrity validation failed")

    def check() -> None:
        events.append("check")
        if events.count("check") == 2:
            raise final_error

    def operation() -> object:
        events.append("operation")
        if outcome == "value_error":
            raise ValueError("PRIVATE")
        if outcome == "runtime_error":
            raise RuntimeError("PRIVATE")
        return object()

    with pytest.raises(TransformValidationError) as caught:
        _delegate_with_binding_guard(check, operation, "source rejected operation")

    assert caught.value is final_error
    assert str(caught.value) == "adapter binding integrity validation failed"
    assert events == ["check", "operation", "check"]


def test_mapping_root_seal_detects_equal_storage_replacement() -> None:
    mapping = _variant().manifest
    seal = _make_mapping_root_seal(mapping)
    original_items = object.__getattribute__(mapping, "_items")
    replacement_items = tuple(list(original_items))
    assert replacement_items == original_items
    assert replacement_items is not original_items

    object.__setattr__(mapping, "_items", replacement_items)

    assert not _mapping_root_seal_matches(seal)


@pytest.mark.parametrize("field_name", ["tools", "manifest"])
def test_schema_runtime_seal_detects_equal_root_replacement(field_name: str) -> None:
    variant = _variant()
    seal = _make_schema_runtime_seal(variant, schema_fingerprint(variant))
    if field_name == "tools":
        replacement = tuple(list(variant.tools))
        assert replacement == variant.tools
        assert replacement is not variant.tools
    else:
        replacement = _variant().manifest
        assert replacement == variant.manifest
        assert replacement is not variant.manifest

    object.__setattr__(variant, field_name, replacement)

    assert not _schema_runtime_seal_matches(seal)


def test_operator_runtime_seal_detects_equal_parameters_root_replacement() -> None:
    operator = _operator()
    seal = _make_operator_runtime_seal(operator)
    replacement = _operator().parameters
    assert replacement == operator.parameters
    assert replacement is not operator.parameters

    object.__setattr__(operator, "parameters", replacement)

    assert not _operator_runtime_seal_matches(seal)


def test_schema_runtime_seal_does_not_scan_nested_schema_nodes() -> None:
    variant = _variant()
    fingerprint = schema_fingerprint(variant)
    properties = variant.tools[0].input_schema["properties"]
    assert isinstance(properties, Mapping)
    query_schema = properties["query"]
    assert isinstance(query_schema, Mapping)
    original_items = object.__getattribute__(query_schema, "_items")
    object.__setattr__(query_schema, "_items", object())

    try:
        seal = _make_schema_runtime_seal(variant, fingerprint)
        assert _schema_runtime_seal_matches(seal)
    finally:
        object.__setattr__(query_schema, "_items", original_items)
