"""Behavior and property tests for deterministic transform manifests and rename."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import toolshift.transforms.base as transform_base_module
import toolshift.transforms.rename as rename_module
from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.schema import schema_fingerprint
from toolshift.transforms.base import (
    TRANSFORM_MANIFEST_SCHEMA_VERSION_HASH,
    OperatorManifestEntry,
    TransformValidationError,
    build_transform_manifest,
    build_transformed_variant,
    derive_operator_seed,
)
from toolshift.transforms.rename import (
    TOOL_NAME_RENAME_VERSION_HASH,
    RenameAdapter,
    RenameTransform,
    apply_rename,
    build_rename_transform,
)
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    canonical_json_bytes,
    manifest_sha256,
)


def _tool(name: str, description: str) -> SurfaceToolSpec:
    return SurfaceToolSpec(
        name,
        description,
        {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "integer", "boolean", "null"]},
            },
        },
    )


def _base_variant(*, reverse_manifest: bool = False) -> SchemaVariant:
    manifest = {"seed": 3, "operator": "identity"}
    if reverse_manifest:
        manifest = {"operator": "identity", "seed": 3}
    return SchemaVariant(
        "base-v1",
        (
            _tool("search", "Search records"),
            _tool("summarize", "Summarize records"),
        ),
        manifest,
    )


def _entry(
    base: SchemaVariant,
    *,
    operator_id: str = "rename-000",
    operator: str = "tool_name_rename",
    level: str = "L1",
    position: int = 0,
    parameters: Mapping[str, JSONValue] | None = None,
) -> OperatorManifestEntry:
    return OperatorManifestEntry(
        operator_id=operator_id,
        operator=operator,
        level=level,
        seed=derive_operator_seed(
            7,
            schema_fingerprint(base),
            operator_id,
            operator,
            position,
        ),
        version_hash=hashlib.sha256(f"toolshift.{operator}.v1".encode()).hexdigest(),
        parameters=parameters or {"mapping": {"tools": {"search": "lookup"}}},
    )


def test_operator_seed_has_a_fixed_domain_separated_vector() -> None:
    assert (
        derive_operator_seed(
            7,
            "a" * 64,
            "rename-000",
            "tool_name_rename",
            0,
        )
        == 2103320145566526
    )
    assert derive_operator_seed(7, "a" * 64, "rename-000", "tool_name_rename", 0) != (
        derive_operator_seed(7, "b" * 64, "rename-000", "tool_name_rename", 0)
    )
    assert derive_operator_seed(7, "a" * 64, "rename-000", "tool_name_rename", 0) != (
        derive_operator_seed(7, "a" * 64, "rename-000", "tool_name_rename", 1)
    )


def test_operator_seed_is_sensitive_to_every_domain_field_and_accepts_boundaries() -> None:
    baseline = derive_operator_seed(7, "a" * 64, "rename-000", "tool_name_rename", 0)
    variants = (
        derive_operator_seed(8, "a" * 64, "rename-000", "tool_name_rename", 0),
        derive_operator_seed(7, "b" * 64, "rename-000", "tool_name_rename", 0),
        derive_operator_seed(7, "a" * 64, "rename-001", "tool_name_rename", 0),
        derive_operator_seed(7, "a" * 64, "rename-000", "docs_format", 0),
        derive_operator_seed(7, "a" * 64, "rename-000", "tool_name_rename", 1),
    )
    assert all(value != baseline for value in variants)
    for boundary in (0, 2**53 - 1):
        derived = derive_operator_seed(
            boundary,
            "a" * 64,
            "rename-000",
            "tool_name_rename",
            boundary,
        )
        assert type(derived) is int
        assert 0 <= derived < 2**53


@pytest.mark.parametrize("seed", [True, False, -1, 2**53, 1.0, "7", None])
def test_operator_seed_rejects_nonexact_or_out_of_range_master_seed(seed: object) -> None:
    with pytest.raises(TransformValidationError, match="seed"):
        derive_operator_seed(seed, "a" * 64, "rename-000", "tool_name_rename", 0)  # type: ignore[arg-type]


@pytest.mark.parametrize("position", [True, False, -1, 2**53, 0.0, "0", None])
def test_operator_seed_rejects_nonexact_or_out_of_range_position(position: object) -> None:
    with pytest.raises(TransformValidationError, match="position"):
        derive_operator_seed(
            7,
            "a" * 64,
            "rename-000",
            "tool_name_rename",
            position,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("fingerprint", "operator_id", "operator"),
    [
        ("A" * 64, "rename-000", "tool_name_rename"),
        ("a" * 63, "rename-000", "tool_name_rename"),
        ("a" * 64, "unsafe/id", "tool_name_rename"),
        ("a" * 64, "rename-000", "unsafe operator"),
    ],
)
def test_operator_seed_rejects_invalid_domain_fields(
    fingerprint: str,
    operator_id: str,
    operator: str,
) -> None:
    with pytest.raises(TransformValidationError):
        derive_operator_seed(7, fingerprint, operator_id, operator, 0)


def test_manifest_entry_deep_snapshots_parameters_and_is_frozen() -> None:
    base = _base_variant()
    parameters: dict[str, JSONValue] = {
        "mapping": {"tools": {"search": "lookup"}},
        "choices": [None, False, 1, {"nested": ["\u03bb"]}],
    }
    entry = _entry(base, parameters=parameters)

    parameters["mapping"] = {"tools": {"search": "changed"}}
    cast_choices = parameters["choices"]
    assert isinstance(cast_choices, list)
    cast_choices.append("changed")

    assert entry.parameters["mapping"]["tools"]["search"] == "lookup"  # type: ignore[index]
    assert entry.parameters["choices"] == (None, False, 1, {"nested": ("\u03bb",)})
    with pytest.raises(TypeError):
        entry.parameters["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        entry.level = "L2"  # type: ignore[misc]


def test_manifest_entry_revalidates_object_setattr_tamper_before_serialization() -> None:
    entry = _entry(_base_variant())
    object.__setattr__(entry, "operator", "docs_format")
    with pytest.raises(TransformValidationError, match="integrity"):
        entry.as_manifest()


def test_manifest_entry_serializes_exact_shape_and_deep_immutable_output() -> None:
    entry = _entry(_base_variant())
    manifest = entry.as_manifest()
    assert set(manifest) == {
        "id",
        "operator",
        "level",
        "seed",
        "version_hash",
        "parameters",
    }
    assert manifest["id"] == entry.operator_id
    assert manifest["operator"] == entry.operator
    assert manifest["level"] == "L1"
    assert manifest["seed"] == entry.seed
    assert manifest["version_hash"] == entry.version_hash
    assert manifest["parameters"] == entry.parameters
    with pytest.raises(TypeError):
        manifest["seed"] = 0  # type: ignore[index]
    parameters = manifest["parameters"]
    assert isinstance(parameters, Mapping)
    mapping = parameters["mapping"]
    assert isinstance(mapping, Mapping)
    tools = mapping["tools"]
    assert isinstance(tools, Mapping)
    with pytest.raises(TypeError):
        tools["search"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("operator_id", "rename-999"),
        ("operator", "docs_format"),
        ("level", "L2"),
        ("seed", 42),
        ("version_hash", "b" * 64),
        ("parameters", {"mapping": {"tools": {"search": "changed"}}}),
        ("_canonical_snapshot", b"changed"),
        ("_snapshot_fingerprint", "b" * 64),
    ],
)
def test_manifest_entry_detects_each_public_and_snapshot_tamper(
    field_name: str,
    replacement: object,
) -> None:
    entry = _entry(_base_variant())
    object.__setattr__(entry, field_name, replacement)
    with pytest.raises(TransformValidationError):
        entry.as_manifest()
    with pytest.raises(TransformValidationError):
        build_transform_manifest(_base_variant(), seed=7, operators=(entry,))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"operator_id": "bad/id"}, "operator_id"),
        ({"operator": "bad operator"}, "operator"),
        ({"level": "L4"}, "level"),
        ({"seed": True}, "seed"),
        ({"seed": 2**53}, "seed"),
        ({"version_hash": "A" * 64}, "version_hash"),
        ({"version_hash": "a" * 63}, "version_hash"),
        ({"parameters": {"bad": float("nan")}}, "parameters"),
        ({"parameters": {1: "bad"}}, "parameters"),
    ],
)
def test_manifest_entry_rejects_invalid_fields(
    overrides: dict[str, object],
    message: str,
) -> None:
    base = _base_variant()
    values: dict[str, object] = {
        "operator_id": "rename-000",
        "operator": "tool_name_rename",
        "level": "L1",
        "seed": derive_operator_seed(
            7,
            schema_fingerprint(base),
            "rename-000",
            "tool_name_rename",
            0,
        ),
        "version_hash": "a" * 64,
        "parameters": {"mapping": {"tools": {"search": "lookup"}}},
    }
    values.update(overrides)
    with pytest.raises(TransformValidationError, match=message):
        OperatorManifestEntry(**values)  # type: ignore[arg-type]


def test_manifest_is_deterministic_ordered_and_non_circular() -> None:
    base = _base_variant()
    first = _entry(base, parameters={"z": 1, "a": {"y": 2, "x": 3}})
    second = _entry(
        base,
        operator_id="docs-001",
        operator="docs_format",
        position=1,
        parameters={"format": "compact"},
    )
    manifest = build_transform_manifest(base, seed=7, operators=(first, second))
    tools = tuple(
        SurfaceToolSpec(
            "lookup" if tool.name == "search" else tool.name,
            tool.description,
            tool.input_schema,
        )
        for tool in base.tools
    )
    variant = build_transformed_variant(
        base,
        tools,
        seed=7,
        operators=(first, second),
    )

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "toolshift_interface_variant"
    assert manifest["base_schema_fingerprint"] == schema_fingerprint(base)
    assert manifest["seed"] == 7
    assert manifest["composition_order"] == ("rename-000", "docs-001")
    assert manifest["version_hash"] == TRANSFORM_MANIFEST_SCHEMA_VERSION_HASH
    assert manifest["operators"][0]["id"] == "rename-000"  # type: ignore[index]
    assert manifest["operators"][1]["id"] == "docs-001"  # type: ignore[index]
    assert set(manifest) == {
        "schema_version",
        "kind",
        "base_schema_fingerprint",
        "seed",
        "composition_order",
        "version_hash",
        "operators",
    }
    assert not {"variant_id", "schema_fingerprint", "manifest_sha256"} & set(manifest)
    assert set(manifest["operators"][0]) == {  # type: ignore[arg-type,index]
        "id",
        "operator",
        "level",
        "seed",
        "version_hash",
        "parameters",
    }
    assert variant.variant_id == f"toolshift-v1-{manifest_sha256(manifest)}"
    assert variant.manifest == manifest

    reordered_base = _base_variant(reverse_manifest=True)
    reordered_first = _entry(
        reordered_base,
        parameters={"a": {"x": 3, "y": 2}, "z": 1},
    )
    reordered_second = _entry(
        reordered_base,
        operator_id="docs-001",
        operator="docs_format",
        position=1,
        parameters={"format": "compact"},
    )
    reordered = build_transform_manifest(
        reordered_base,
        seed=7,
        operators=(reordered_first, reordered_second),
    )
    assert canonical_json_bytes(reordered) == canonical_json_bytes(manifest)
    with pytest.raises(TypeError):
        manifest["seed"] = 8  # type: ignore[index]
    operators = manifest["operators"]
    assert isinstance(operators, tuple)
    with pytest.raises(TypeError):
        operators[0]["operator"] = "changed"  # type: ignore[index]


def test_reversing_composition_with_rederived_seeds_changes_order_and_digest() -> None:
    base = _base_variant()
    rename_first = _entry(base, position=0)
    docs_second = _entry(
        base,
        operator_id="docs-001",
        operator="docs_format",
        position=1,
        parameters={"format": "compact"},
    )
    docs_first = _entry(
        base,
        operator_id="docs-001",
        operator="docs_format",
        position=0,
        parameters={"format": "compact"},
    )
    rename_second = _entry(base, position=1)
    forward = build_transform_manifest(
        base,
        seed=7,
        operators=(rename_first, docs_second),
    )
    reverse = build_transform_manifest(
        base,
        seed=7,
        operators=(docs_first, rename_second),
    )
    forward_variant = build_transformed_variant(
        base,
        base.tools,
        seed=7,
        operators=(rename_first, docs_second),
    )
    reverse_variant = build_transformed_variant(
        base,
        base.tools,
        seed=7,
        operators=(docs_first, rename_second),
    )
    assert forward["composition_order"] == ("rename-000", "docs-001")
    assert reverse["composition_order"] == ("docs-001", "rename-000")
    assert manifest_sha256(forward) != manifest_sha256(reverse)
    assert forward_variant.variant_id != reverse_variant.variant_id


def test_built_manifest_is_a_snapshot_of_entry_and_source_constructor_inputs() -> None:
    base_manifest: dict[str, JSONValue] = {"operator": "identity", "seed": 3}
    base = SchemaVariant("base-snapshot-v1", _base_variant().tools, base_manifest)
    parameters: dict[str, JSONValue] = {"mapping": {"tools": {"search": "lookup"}}}
    entry = _entry(base, parameters=parameters)
    manifest = build_transform_manifest(base, seed=7, operators=(entry,))
    before = canonical_json_bytes(manifest)
    base_manifest["seed"] = 999
    parameters["mapping"] = {"tools": {"search": "changed"}}
    object.__setattr__(entry, "parameters", {"changed": True})
    assert canonical_json_bytes(manifest) == before
    with pytest.raises(TransformValidationError):
        build_transform_manifest(base, seed=7, operators=(entry,))


def test_manifest_schema_version_hash_is_a_fixed_content_identifier() -> None:
    assert (
        hashlib.sha256(b"toolshift.transform-manifest.schema.v1").hexdigest()
        == TRANSFORM_MANIFEST_SCHEMA_VERSION_HASH
    )
    base = _base_variant()
    manifest = build_transform_manifest(base, seed=7, operators=(_entry(base),))
    assert manifest["version_hash"] != manifest_sha256(manifest)


def test_tool_name_rename_abi_hash_locks_the_declared_content_tag() -> None:
    assert (
        hashlib.sha256(b"toolshift.transform.tool-name-rename.v1").hexdigest()
        == TOOL_NAME_RENAME_VERSION_HASH
    )
    assert TOOL_NAME_RENAME_VERSION_HASH == (
        "3f5bb2bfbc1c6926f8e1da5096a3a50375eb34bf941c32dd3b7ab7f90d135510"
    )


def test_manifest_rejects_empty_non_tuple_duplicate_and_wrong_position_seed() -> None:
    base = _base_variant()
    entry = _entry(base)
    with pytest.raises(TransformValidationError, match="non-empty tuple"):
        build_transform_manifest(base, seed=7, operators=())
    with pytest.raises(TransformValidationError, match="exact tuple"):
        build_transform_manifest(base, seed=7, operators=[entry])  # type: ignore[arg-type]
    duplicate = _entry(base, position=1)
    with pytest.raises(TransformValidationError, match="unique"):
        build_transform_manifest(base, seed=7, operators=(entry, duplicate))
    wrong_seed = OperatorManifestEntry(
        entry.operator_id,
        entry.operator,
        entry.level,
        entry.seed + 1,
        entry.version_hash,
        entry.parameters,
    )
    with pytest.raises(TransformValidationError, match="derived"):
        build_transform_manifest(base, seed=7, operators=(wrong_seed,))


def test_build_transformed_variant_requires_exact_tools_tuple() -> None:
    base = _base_variant()
    entry = _entry(base)
    with pytest.raises(TransformValidationError, match="tools"):
        build_transformed_variant(base, list(base.tools), seed=7, operators=(entry,))  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [True, -1, 2**53, 1.0])
def test_public_manifest_builder_rejects_invalid_master_seed(seed: object) -> None:
    base = _base_variant()
    with pytest.raises(TransformValidationError, match="seed"):
        build_transform_manifest(base, seed=seed, operators=(_entry(base),))  # type: ignore[arg-type]


def test_public_builders_reject_wrong_operator_and_tool_entry_types() -> None:
    base = _base_variant()
    with pytest.raises(TransformValidationError, match="OperatorManifestEntry"):
        build_transform_manifest(base, seed=7, operators=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TransformValidationError, match="tools"):
        build_transformed_variant(
            base,
            (object(),),  # type: ignore[arg-type]
            seed=7,
            operators=(_entry(base),),
        )


def test_transformed_variant_rejects_object_setattr_mutated_exact_tool() -> None:
    base = _base_variant()
    damaged = _tool("lookup", "Damaged tool")
    object.__setattr__(damaged, "name", "")
    with pytest.raises(TransformValidationError, match="valid schema"):
        build_transformed_variant(
            base,
            (damaged,),
            seed=7,
            operators=(_entry(base),),
        )


def test_partial_rename_preserves_every_non_name_tool_field_and_records_full_map() -> None:
    base = _base_variant()
    supplied = {"search": "lookup"}
    transform = build_rename_transform(base, tool_name_mapping=supplied, seed=7)
    assert supplied == {"search": "lookup"}
    supplied["search"] = "changed"

    assert isinstance(transform, RenameTransform)
    assert transform.source_variant is base
    assert tuple(tool.name for tool in transform.variant.tools) == ("lookup", "summarize")
    for original, shifted in zip(base.tools, transform.variant.tools, strict=True):
        assert shifted.description == original.description
        assert shifted.input_schema == original.input_schema
    assert dict(transform.canonical_to_surface) == {
        "search": "lookup",
        "summarize": "summarize",
    }
    operator_manifest = transform.variant.manifest["operators"][0]  # type: ignore[index]
    assert operator_manifest["operator"] == "tool_name_rename"  # type: ignore[index]
    assert operator_manifest["level"] == "L1"  # type: ignore[index]
    assert operator_manifest["version_hash"] == TOOL_NAME_RENAME_VERSION_HASH  # type: ignore[index]
    assert operator_manifest["parameters"]["mapping"]["tools"] == {  # type: ignore[index]
        "search": "lookup",
        "summarize": "summarize",
    }
    with pytest.raises(TypeError):
        transform.canonical_to_surface["search"] = "changed"  # type: ignore[index]


def test_multi_rename_is_order_invariant_accepts_generic_mapping_and_preserves_unicode() -> None:
    base = SchemaVariant(
        "three-tool-base-v1",
        (*_base_variant().tools, _tool("translate", "Translate records")),
        {"operator": "identity", "seed": 3},
    )
    first_input = MappingProxyType({"search": "\u641c\u7d22", "translate": "e\u0301"})
    second_input = MappingProxyType({"translate": "e\u0301", "search": "\u641c\u7d22"})
    first = build_rename_transform(base, tool_name_mapping=first_input, seed=7)
    second = build_rename_transform(base, tool_name_mapping=second_input, seed=7)
    assert tuple(tool.name for tool in first.variant.tools) == (
        "\u641c\u7d22",
        "summarize",
        "e\u0301",
    )
    assert canonical_json_bytes(first.variant.manifest) == canonical_json_bytes(
        second.variant.manifest
    )
    assert first.variant.variant_id == second.variant.variant_id
    composed = build_rename_transform(
        base,
        tool_name_mapping={"search": "\u00e9", "translate": "e\u0301"},
        seed=7,
    )
    assert tuple(tool.name for tool in composed.variant.tools) == (
        "\u00e9",
        "summarize",
        "e\u0301",
    )


class _DuplicateItemsMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        if key != "search":
            raise KeyError(key)
        return "lookup"

    def __iter__(self):
        return iter(("search",))

    def __len__(self) -> int:
        return 1

    def items(self):
        return (("search", "lookup"), ("search", "changed"))


class _DuplicateFullItemsMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        return {"search": "lookup", "summarize": "summarize"}[key]

    def __iter__(self):
        return iter(("search", "summarize"))

    def __len__(self) -> int:
        return 2

    def items(self):
        return (
            ("search", "lookup"),
            ("search", "changed"),
            ("summarize", "summarize"),
        )


def test_rename_rejects_callback_mapping_with_duplicate_canonical_entries() -> None:
    with pytest.raises(TransformValidationError, match="duplicate"):
        build_rename_transform(
            _base_variant(),
            tool_name_mapping=_DuplicateItemsMapping(),
            seed=7,
        )


def test_direct_transform_rejects_duplicate_full_mapping_before_freezing() -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    with pytest.raises(TransformValidationError, match="duplicate"):
        RenameTransform(
            transform.source_variant,
            transform.variant,
            transform.operator,
            _DuplicateFullItemsMapping(),
        )


def test_same_surface_mapping_under_different_master_seeds_has_distinct_manifest_ids() -> None:
    base = _base_variant()
    first = build_rename_transform(base, tool_name_mapping={"search": "lookup"}, seed=7)
    second = build_rename_transform(base, tool_name_mapping={"search": "lookup"}, seed=8)
    assert tuple(tool.name for tool in first.variant.tools) == tuple(
        tool.name for tool in second.variant.tools
    )
    assert first.variant.variant_id != second.variant.variant_id
    assert first.variant.manifest != second.variant.manifest


@pytest.mark.parametrize(
    "mapping",
    [
        {},
        {"missing": "lookup"},
        {"search": "search"},
        {"search": "lookup", "summarize": "summarize"},
        {"search": "summarize"},
        {"search": "lookup", "summarize": "lookup"},
        {"search": "summarize", "summarize": "search"},
        {"search": ""},
        {"search": "   "},
        {"search": "bad\ud800"},
        {1: "lookup"},
        {"search": 1},
        None,
        [],
    ],
)
def test_rename_rejects_empty_unknown_noop_colliding_or_invalid_mapping(mapping: object) -> None:
    with pytest.raises(TransformValidationError):
        build_rename_transform(
            _base_variant(),
            tool_name_mapping=mapping,  # type: ignore[arg-type]
            seed=7,
        )


_SAFE_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    max_size=20,
)
_SAFE_KEY = _SAFE_TEXT
_JSON_SCALAR = st.none() | st.booleans() | st.integers(-(2**53 - 1), 2**53 - 1) | _SAFE_TEXT
_JSON_VALUE = st.recursive(
    _JSON_SCALAR,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(_SAFE_KEY, children, max_size=4)
    ),
    max_leaves=20,
)


@settings(max_examples=80, deadline=None)
@given(
    arguments=st.dictionaries(_SAFE_KEY, _JSON_VALUE, max_size=6),
    metadata=_JSON_VALUE,
)
def test_rename_call_round_trip_preserves_arbitrary_ijson(
    arguments: dict[str, JSONValue],
    metadata: JSONValue,
) -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    surface_call: dict[str, JSONValue] = {
        "metadata": metadata,
        "arguments": arguments,
        "name": "lookup",
    }

    canonical = transform.surface_call_to_canonical(surface_call)
    assert canonical["name"] == "search"
    restored = transform.canonical_call_to_surface(canonical)
    assert canonical_json_bytes(restored) == canonical_json_bytes(surface_call)
    assert canonical_json_bytes(
        transform.surface_call_to_canonical(restored)
    ) == canonical_json_bytes(canonical)


def test_rename_call_canonicalization_ignores_mapping_key_order_but_not_json_types() -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    left = {"name": "lookup", "arguments": {"b": 1, "a": [True, None]}}
    right = {"arguments": {"a": [True, None], "b": 1}, "name": "lookup"}
    integer = {"name": "lookup", "arguments": {"value": 1}}
    boolean = {"name": "lookup", "arguments": {"value": True}}
    floating = {"name": "lookup", "arguments": {"value": 1.0}}
    assert canonical_json_bytes(transform.surface_call_to_canonical(left)) == (
        canonical_json_bytes(transform.surface_call_to_canonical(right))
    )
    assert canonical_json_bytes(transform.surface_call_to_canonical(integer)) != (
        canonical_json_bytes(transform.surface_call_to_canonical(boolean))
    )
    assert canonical_json_bytes(transform.surface_call_to_canonical(integer)) != (
        canonical_json_bytes(transform.surface_call_to_canonical(floating))
    )
    assert canonical_json_bytes(transform.surface_call_to_canonical(boolean)) != (
        canonical_json_bytes(transform.surface_call_to_canonical(floating))
    )
    assert (
        transform.surface_call_to_canonical({"name": "lookup", "arguments": {"value": 1.5}})[
            "arguments"
        ]["value"]
        == 1.5
    )  # type: ignore[index]
    with pytest.raises(TransformValidationError):
        transform.surface_call_to_canonical(
            {"name": "lookup", "arguments": {"value": float("inf")}}
        )


def test_call_translation_snapshots_without_mutating_inputs_and_deep_freezes_outputs() -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    nested: list[JSONValue] = [1, {"value": [True, None]}]
    call: dict[str, JSONValue] = {"name": "lookup", "arguments": {"nested": nested}}
    before = canonical_json_bytes(call)
    canonical = transform.surface_call_to_canonical(call)
    assert canonical_json_bytes(call) == before
    assert canonical is not call
    nested.append("changed")
    assert canonical["arguments"]["nested"] == (1, {"value": (True, None)})  # type: ignore[index]
    with pytest.raises(TypeError):
        canonical["name"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "call",
    [
        {"name": "search", "arguments": {}},
        {"name": "unknown", "arguments": {"private": "DO_NOT_LEAK"}},
        {"name": 7, "arguments": {}},
        {"arguments": {}},
        {"name": "lookup", "arguments": {"bad": object()}},
    ],
)
def test_surface_call_translation_rejects_nonfinal_or_malformed_calls_payload_free(
    call: dict[str, object],
) -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    with pytest.raises(TransformValidationError) as caught:
        transform.surface_call_to_canonical(call)  # type: ignore[arg-type]
    assert "DO_NOT_LEAK" not in str(caught.value)
    assert "object" not in str(caught.value)


class _ExplodingCallMapping(Mapping[str, JSONValue]):
    def __getitem__(self, key: str) -> JSONValue:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        raise TransformValidationError("SECRET_CALL_CALLBACK_PAYLOAD")


class _ExplodingOperatorParameters(Mapping[str, JSONValue]):
    def __getitem__(self, key: str) -> JSONValue:
        raise KeyError(key)

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self):
        raise ValueError("PRIVATE_OPERATOR_PAYLOAD")


class _ExplodingVariantManifest(Mapping[str, JSONValue]):
    def __getitem__(self, key: str) -> JSONValue:
        raise TransformValidationError("SECRET_VARIANT_MANIFEST_PAYLOAD")

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0


class _ExplodingTools(tuple[SurfaceToolSpec, ...]):
    def __iter__(self):
        raise TransformValidationError("SECRET_SOURCE_TOOLS_PAYLOAD")


class _MutatingItemsMapping(Mapping[str, JSONValue]):
    def __init__(
        self,
        entries: Mapping[str, JSONValue],
        mutation: Callable[[], None],
    ) -> None:
        self._entries = dict(entries)
        self._mutation = mutation

    def __getitem__(self, key: str) -> JSONValue:
        return self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self):
        self._mutation()
        return self._entries.items()


def test_call_mapping_callback_cannot_smuggle_payload_through_transform_error_type() -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    with pytest.raises(TransformValidationError) as caught:
        transform.surface_call_to_canonical(_ExplodingCallMapping())
    assert "SECRET_CALL_CALLBACK_PAYLOAD" not in str(caught.value)


def test_direct_transform_constructor_sanitizes_tampered_operator_parameters() -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    object.__setattr__(transform.operator, "parameters", _ExplodingOperatorParameters())
    with pytest.raises(TransformValidationError) as caught:
        RenameTransform(
            transform.source_variant,
            transform.variant,
            transform.operator,
            transform.canonical_to_surface,
        )
    assert "PRIVATE_OPERATOR_PAYLOAD" not in str(caught.value)


@pytest.mark.parametrize("target", ["variant_manifest", "source_tools"])
def test_direct_transform_constructor_sanitizes_tampered_variant_callbacks(
    target: str,
) -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    if target == "variant_manifest":
        object.__setattr__(transform.variant, "manifest", _ExplodingVariantManifest())
        secret = "SECRET_VARIANT_MANIFEST_PAYLOAD"
    else:
        object.__setattr__(
            transform.source_variant,
            "tools",
            _ExplodingTools(transform.source_variant.tools),
        )
        secret = "SECRET_SOURCE_TOOLS_PAYLOAD"

    with pytest.raises(TransformValidationError) as caught:
        RenameTransform(
            transform.source_variant,
            transform.variant,
            transform.operator,
            transform.canonical_to_surface,
        )
    assert secret not in str(caught.value)


@pytest.mark.parametrize("target", ["source_tools", "operator_parameters"])
def test_direct_transform_snapshots_external_mapping_before_deep_validation(
    target: str,
) -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    if target == "source_tools":
        secret = "SECRET_SOURCE_TOOLS_PAYLOAD"

        def mutate() -> None:
            object.__setattr__(
                transform.source_variant,
                "tools",
                _ExplodingTools(transform.source_variant.tools),
            )

    else:
        secret = "PRIVATE_OPERATOR_PAYLOAD"

        def mutate() -> None:
            object.__setattr__(
                transform.operator,
                "parameters",
                _ExplodingOperatorParameters(),
            )

    mapping = _MutatingItemsMapping(transform.canonical_to_surface, mutate)
    with pytest.raises(TransformValidationError) as caught:
        RenameTransform(
            transform.source_variant,
            transform.variant,
            transform.operator,
            mapping,
        )
    assert secret not in str(caught.value)


def test_direct_transform_exhausts_operator_callback_before_variant_validation() -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )

    def mutate() -> None:
        object.__setattr__(
            transform.source_variant,
            "tools",
            _ExplodingTools(transform.source_variant.tools),
        )

    object.__setattr__(
        transform.operator,
        "parameters",
        _MutatingItemsMapping(transform.operator.parameters, mutate),
    )
    with pytest.raises(TransformValidationError) as caught:
        RenameTransform(
            transform.source_variant,
            transform.variant,
            transform.operator,
            transform.canonical_to_surface,
        )
    assert "SECRET_SOURCE_TOOLS_PAYLOAD" not in str(caught.value)


def test_public_rename_builder_revalidates_base_after_mapping_callback() -> None:
    base = _base_variant()

    def mutate() -> None:
        object.__setattr__(base, "tools", _ExplodingTools(base.tools))

    mapping = _MutatingItemsMapping({"search": "lookup"}, mutate)
    with pytest.raises(TransformValidationError) as caught:
        build_rename_transform(
            base,
            tool_name_mapping=mapping,  # type: ignore[arg-type]
            seed=7,
        )
    assert "SECRET_SOURCE_TOOLS_PAYLOAD" not in str(caught.value)


def test_manifest_entry_resnapshots_parameters_before_rechecking_other_fields() -> None:
    entry = _entry(_base_variant())

    def mutate() -> None:
        object.__setattr__(entry, "operator", "docs_format")

    parameters = _MutatingItemsMapping(entry.parameters, mutate)
    object.__setattr__(entry, "parameters", parameters)
    with pytest.raises(TransformValidationError, match="integrity"):
        entry.as_manifest()


def test_manifest_entry_constructor_exhausts_parameter_callback_before_validation() -> None:
    entry = object.__new__(OperatorManifestEntry)

    def mutate() -> None:
        object.__setattr__(entry, "level", "L4")

    parameters = _MutatingItemsMapping({"mode": "rename"}, mutate)
    with pytest.raises(TransformValidationError, match="level"):
        OperatorManifestEntry.__init__(
            entry,
            "rename-000",
            "tool_name_rename",
            "L1",
            0,
            "a" * 64,
            parameters,
        )


@pytest.mark.parametrize("builder", ["manifest", "variant"])
def test_foundation_revalidates_base_after_operator_parameter_callback(
    builder: str,
) -> None:
    base = _base_variant()
    original_tools = base.tools
    entry = _entry(base)

    def mutate() -> None:
        object.__setattr__(base, "tools", _ExplodingTools(base.tools))

    object.__setattr__(
        entry,
        "parameters",
        _MutatingItemsMapping(entry.parameters, mutate),
    )
    with pytest.raises(TransformValidationError) as caught:
        if builder == "manifest":
            build_transform_manifest(base, seed=7, operators=(entry,))
        else:
            build_transformed_variant(
                base,
                original_tools,
                seed=7,
                operators=(entry,),
            )
    assert "SECRET_SOURCE_TOOLS_PAYLOAD" not in str(caught.value)


def test_transformed_variant_rejects_callback_bearing_corrupted_tool_schema() -> None:
    base = _base_variant()
    entry = _entry(base)
    first_tool = SurfaceToolSpec(
        base.tools[0].name,
        base.tools[0].description,
        base.tools[0].input_schema,
    )

    def mutate() -> None:
        object.__setattr__(base, "tools", _ExplodingTools(base.tools))

    object.__setattr__(
        first_tool,
        "input_schema",
        _MutatingItemsMapping(first_tool.input_schema, mutate),
    )
    with pytest.raises(TransformValidationError) as caught:
        build_transformed_variant(
            base,
            (first_tool, base.tools[1]),
            seed=7,
            operators=(entry,),
        )
    assert "SECRET_SOURCE_TOOLS_PAYLOAD" not in str(caught.value)


def test_canonical_call_translation_rejects_alias_and_unknown_names() -> None:
    transform = build_rename_transform(
        _base_variant(),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    for name in ("lookup", "unknown"):
        with pytest.raises(TransformValidationError, match="canonical"):
            transform.canonical_call_to_surface({"name": name, "arguments": {}})


class _RecordingIdentityAdapter(SemanticAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self.parse_inputs: list[Mapping[str, JSONValue]] = []
        self.parse_results: list[tuple[SemanticAction, ...]] = []
        self.compile_inputs: list[SemanticAction] = []
        self.compile_results: list[tuple[Mapping[str, JSONValue], ...]] = []
        self.wrap_inputs: list[
            tuple[
                Mapping[str, JSONValue],
                tuple[SemanticAction, ...],
                tuple[tuple[JSONValue, ...], ...],
            ]
        ] = []
        self.trace_inputs: list[ExecutionTrace] = []

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        self.parse_inputs.append(surface_call)
        name = surface_call.get("name")
        arguments = surface_call.get("arguments")
        if type(name) is not str or name not in {tool.name for tool in self.variant.tools}:
            raise ValueError("source parse payload PRIVATE_PARSE")
        if not isinstance(arguments, Mapping):
            raise ValueError("source parse payload PRIVATE_PARSE")
        result = (SemanticAction(name, arguments),)
        self.parse_results.append(result)
        return result

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self.compile_inputs.append(action)
        result = ({"name": action.name, "arguments": action.arguments},)
        self.compile_results.append(result)
        return result

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self.wrap_inputs.append((surface_call, actions, base_observation_groups))
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self.trace_inputs.append(trace)
        return trace.semantic_actions


class _PayloadPublicVariantAdapter(_RecordingIdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        self.raise_public = False
        super().__init__(variant)

    @property
    def variant(self) -> SchemaVariant:
        if self.raise_public:
            raise TransformValidationError("SECRET_BINDING_CALLBACK_PAYLOAD")
        return super().variant


class _WrongBoundSourceAdapter(_RecordingIdentityAdapter):
    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        arguments = surface_call.get("arguments")
        assert isinstance(arguments, Mapping)
        return (SemanticAction("summarize", arguments),)


def test_rename_adapter_delegates_the_actual_pipeline_objects_exactly_once() -> None:
    source = _RecordingIdentityAdapter(_base_variant())
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    assert isinstance(adapter, RenameAdapter)
    surface_call = {"name": "lookup", "arguments": {"value": [True, 1, None]}}

    actions = adapter.surface_to_semantic(surface_call)
    assert actions is source.parse_results[0]
    assert actions[0] is source.parse_results[0][0]
    assert source.parse_inputs[0]["name"] == "search"

    base_calls = adapter.semantic_to_base_calls(actions[0])
    assert source.compile_inputs == [actions[0]]
    assert base_calls is source.compile_results[0]

    observation: JSONValue = {"items": [1, False, None]}
    groups = ((observation,),)
    wrapped = adapter.base_observation_to_surface(surface_call, actions, groups)
    delegated_call, delegated_actions, delegated_groups = source.wrap_inputs[0]
    assert delegated_call["name"] == "search"
    assert delegated_actions is actions
    assert delegated_groups is groups
    assert wrapped is observation
    assert len(source.parse_inputs) == len(source.compile_inputs) == len(source.wrap_inputs) == 1


def test_large_schema_online_adapter_never_rebuilds_or_traverses_schema_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = {f"field_{index}": {"type": "string"} for index in range(50)}
    base = SchemaVariant(
        "large-base-v1",
        tuple(
            SurfaceToolSpec(
                f"tool_{index}",
                f"Tool {index}",
                {"type": "object", "properties": properties},
            )
            for index in range(100)
        ),
        {"operator": "identity", "seed": 0},
    )
    adapter = apply_rename(
        _RecordingIdentityAdapter(base),
        tool_name_mapping={"tool_0": "renamed_0"},
        seed=7,
    )

    def unexpected_deep_validation(*args: object, **kwargs: object) -> object:
        pytest.fail("online adapter invoked construction-time deep schema validation")

    monkeypatch.setattr(rename_module, "schema_fingerprint", unexpected_deep_validation)
    monkeypatch.setattr(rename_module, "build_transformed_variant", unexpected_deep_validation)
    monkeypatch.setattr(rename_module, "canonical_json_bytes", unexpected_deep_validation)
    monkeypatch.setattr(rename_module, "_make_schema_runtime_seal", unexpected_deep_validation)
    monkeypatch.setattr(rename_module, "_make_operator_runtime_seal", unexpected_deep_validation)
    monkeypatch.setattr(rename_module, "_make_mapping_root_seal", unexpected_deep_validation)
    monkeypatch.setattr(transform_base_module, "schema_fingerprint", unexpected_deep_validation)
    for value in range(5):
        actions = adapter.surface_to_semantic(
            {"name": "renamed_0", "arguments": {"field_0": str(value)}}
        )
        adapter.semantic_to_base_calls(actions[0])


def test_public_variant_callback_payload_is_sanitized_at_init_and_runtime() -> None:
    initial_source = _PayloadPublicVariantAdapter(_base_variant())
    initial_source.raise_public = True
    with pytest.raises(TransformValidationError) as initial_error:
        apply_rename(
            initial_source,
            tool_name_mapping={"search": "lookup"},
            seed=7,
        )
    assert "SECRET_BINDING_CALLBACK_PAYLOAD" not in str(initial_error.value)

    runtime_source = _PayloadPublicVariantAdapter(_base_variant())
    adapter = apply_rename(
        runtime_source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    runtime_source.raise_public = True
    with pytest.raises(TransformValidationError) as runtime_error:
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})
    assert "SECRET_BINDING_CALLBACK_PAYLOAD" not in str(runtime_error.value)


def test_runtime_guard_rejects_replacing_source_adapter_with_same_bound_variant() -> None:
    base = _base_variant()
    adapter = apply_rename(
        _RecordingIdentityAdapter(base),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    object.__setattr__(adapter, "_source_adapter", _WrongBoundSourceAdapter(base))
    with pytest.raises(TransformValidationError, match="binding"):
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})


def test_apply_rename_rejects_implicit_adapter_stacking_without_compose_contract() -> None:
    first = apply_rename(
        _RecordingIdentityAdapter(_base_variant()),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    with pytest.raises(TransformValidationError, match="compos"):
        apply_rename(
            first,
            tool_name_mapping={"lookup": "find"},
            seed=8,
        )


def test_direct_transform_constructor_rejects_manual_nested_rename() -> None:
    first = apply_rename(
        _RecordingIdentityAdapter(_base_variant()),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    seed = 8
    operator_id = "rename-outer"
    mapping = {"lookup": "find", "summarize": "summarize"}
    operator = OperatorManifestEntry(
        operator_id,
        "tool_name_rename",
        "L1",
        derive_operator_seed(
            seed,
            schema_fingerprint(first.variant),
            operator_id,
            "tool_name_rename",
            0,
        ),
        TOOL_NAME_RENAME_VERSION_HASH,
        {"mapping": {"tools": mapping}},
    )
    variant = build_transformed_variant(
        first.variant,
        (
            SurfaceToolSpec(
                "find",
                first.variant.tools[0].description,
                first.variant.tools[0].input_schema,
            ),
            first.variant.tools[1],
        ),
        seed=seed,
        operators=(operator,),
    )
    with pytest.raises(TransformValidationError, match="compos"):
        RenameTransform(first.variant, variant, operator, mapping)


def test_rename_adapter_constructor_rejects_nested_source_adapter_directly() -> None:
    base = _base_variant()
    transform = build_rename_transform(
        base,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    nested_source = apply_rename(
        _RecordingIdentityAdapter(base),
        tool_name_mapping={"search": "find"},
        seed=8,
    )
    with pytest.raises(TransformValidationError, match="compos"):
        RenameAdapter(nested_source, transform)


def test_public_adapter_rejects_renamed_away_canonical_but_accepts_final_identity() -> None:
    source = _RecordingIdentityAdapter(_base_variant())
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    old_call = {"name": "search", "arguments": {}}
    with pytest.raises(TransformValidationError, match="surface"):
        adapter.surface_to_semantic(old_call)
    with pytest.raises(TransformValidationError, match="surface"):
        adapter.base_observation_to_surface(
            old_call,
            (SemanticAction("search", {}),),
            (({"ok": True},),),
        )

    identity_call = {"name": "summarize", "arguments": {"value": "alpha"}}
    identity_actions = adapter.surface_to_semantic(identity_call)
    assert identity_actions[0].name == "summarize"
    observation: JSONValue = {"summary": "alpha"}
    assert (
        adapter.base_observation_to_surface(
            identity_call,
            identity_actions,
            ((observation,),),
        )
        is observation
    )


def test_rename_adapter_canonicalizes_reference_and_candidate_surface_channels_only() -> None:
    source = _RecordingIdentityAdapter(_base_variant())
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    action = SemanticAction("search", {"value": "alpha"})
    reference = ExecutionTrace(
        ({"name": "search", "arguments": {"value": "alpha"}},),
        (action,),
        ({"name": "search", "arguments": {"value": "alpha"}},),
    )
    candidate = ExecutionTrace(
        ({"name": "lookup", "arguments": {"value": "alpha"}},),
        (action,),
        ({"name": "search", "arguments": {"value": "alpha"}},),
    )

    reference_result = adapter.canonicalize_trace(reference)
    candidate_result = adapter.canonicalize_trace(candidate)

    assert reference_result is reference.semantic_actions
    assert candidate_result is candidate.semantic_actions
    assert source.trace_inputs[0] is reference
    delegated_candidate = source.trace_inputs[1]
    assert delegated_candidate.surface_calls[0]["name"] == "search"
    assert delegated_candidate.semantic_actions is candidate.semantic_actions
    assert delegated_candidate.semantic_actions[0] is candidate.semantic_actions[0]
    assert canonical_json_bytes(delegated_candidate.base_calls) == canonical_json_bytes(
        candidate.base_calls
    )
    unknown = ExecutionTrace(
        ({"name": "unknown", "arguments": {}},),
        (),
        (),
    )
    with pytest.raises(TransformValidationError, match="trace"):
        adapter.canonicalize_trace(unknown)

    hybrid = ExecutionTrace(
        (
            {"name": "search", "arguments": {}},
            {"name": "lookup", "arguments": {}},
        ),
        (action, action),
        (
            {"name": "search", "arguments": {}},
            {"name": "search", "arguments": {}},
        ),
    )
    with pytest.raises(TransformValidationError, match="trace"):
        adapter.canonicalize_trace(hybrid)


class _SourceValueErrorAdapter(_RecordingIdentityAdapter):
    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        raise ValueError("SECRET_SOURCE_PAYLOAD")


class _SourceRuntimeErrorAdapter(_RecordingIdentityAdapter):
    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        raise RuntimeError("unexpected source runtime")


class _StageErrorSourceAdapter(_RecordingIdentityAdapter):
    def __init__(self, variant: SchemaVariant, stage: str, error_type: type[Exception]) -> None:
        super().__init__(variant)
        self._stage = stage
        self._error_type = error_type
        self.error = error_type("PRIVATE_STAGE_PAYLOAD")

    def _maybe_raise(self, stage: str) -> None:
        if self._stage == stage:
            raise self.error

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        self._maybe_raise("parse")
        return super().surface_to_semantic(surface_call)

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self._maybe_raise("compile")
        return super().semantic_to_base_calls(action)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self._maybe_raise("wrap")
        return super().base_observation_to_surface(
            surface_call,
            actions,
            base_observation_groups,
        )

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self._maybe_raise("canonicalize")
        return super().canonicalize_trace(trace)


def test_source_value_errors_are_wrapped_without_payload_or_cause() -> None:
    adapter = apply_rename(
        _SourceValueErrorAdapter(_base_variant()),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    with pytest.raises(TransformValidationError) as caught:
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})
    assert "SECRET_SOURCE_PAYLOAD" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_unexpected_source_exceptions_propagate_unchanged() -> None:
    adapter = apply_rename(
        _SourceRuntimeErrorAdapter(_base_variant()),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    with pytest.raises(RuntimeError, match="unexpected source runtime"):
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})


@pytest.mark.parametrize("stage", ["parse", "compile", "wrap", "canonicalize"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_non_rebinding_source_errors_are_handled_at_every_delegation_stage(
    stage: str,
    error_type: type[Exception],
) -> None:
    source = _StageErrorSourceAdapter(_base_variant(), stage, error_type)
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    action = SemanticAction("search", {})

    def invoke() -> object:
        if stage == "parse":
            return adapter.surface_to_semantic({"name": "lookup", "arguments": {}})
        if stage == "compile":
            return adapter.semantic_to_base_calls(action)
        if stage == "wrap":
            return adapter.base_observation_to_surface(
                {"name": "lookup", "arguments": {}},
                (action,),
                (({"ok": True},),),
            )
        return adapter.canonicalize_trace(
            ExecutionTrace(
                ({"name": "lookup", "arguments": {}},),
                (action,),
                ({"name": "search", "arguments": {}},),
            )
        )

    if error_type is ValueError:
        with pytest.raises(TransformValidationError) as caught:
            invoke()
        assert "PRIVATE_STAGE_PAYLOAD" not in str(caught.value)
        assert caught.value.__cause__ is None
    else:
        with pytest.raises(RuntimeError) as caught:
            invoke()
        assert caught.value is source.error


class _RebindingSourceAdapter(_RecordingIdentityAdapter):
    def __init__(self, variant: SchemaVariant, stage: str) -> None:
        super().__init__(variant)
        self._stage = stage
        self._replacement = SchemaVariant(
            "replacement-v1",
            variant.tools,
            {"operator": "replacement", "seed": 9},
        )

    def _rebind(self, stage: str) -> None:
        if self._stage == stage:
            self._variant = self._replacement

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        result = super().surface_to_semantic(surface_call)
        self._rebind("parse")
        return result

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        result = super().semantic_to_base_calls(action)
        self._rebind("compile")
        return result

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        result = super().base_observation_to_surface(
            surface_call,
            actions,
            base_observation_groups,
        )
        self._rebind("wrap")
        return result

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        result = super().canonicalize_trace(trace)
        self._rebind("canonicalize")
        return result


class _RebindThenRaiseSourceAdapter(_RebindingSourceAdapter):
    def __init__(self, variant: SchemaVariant, stage: str, error_type: type[Exception]) -> None:
        super().__init__(variant, stage)
        self._error_type = error_type

    def _raise_after_rebind(self, stage: str) -> None:
        self._rebind(stage)
        raise self._error_type("PRIVATE_EXCEPTION_PAYLOAD")

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        self._raise_after_rebind("parse")

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self._raise_after_rebind("compile")

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self._raise_after_rebind("wrap")

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self._raise_after_rebind("canonicalize")


@pytest.mark.parametrize("stage", ["parse", "compile", "wrap", "canonicalize"])
def test_rename_adapter_detects_source_rebinding_after_every_delegation(stage: str) -> None:
    source = _RebindingSourceAdapter(_base_variant(), stage)
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    action = SemanticAction("search", {})
    with pytest.raises(TransformValidationError, match="binding"):
        if stage == "parse":
            adapter.surface_to_semantic({"name": "lookup", "arguments": {}})
        elif stage == "compile":
            adapter.semantic_to_base_calls(action)
        elif stage == "wrap":
            adapter.base_observation_to_surface(
                {"name": "lookup", "arguments": {}},
                (action,),
                (({"ok": True},),),
            )
        else:
            adapter.canonicalize_trace(
                ExecutionTrace(
                    ({"name": "lookup", "arguments": {}},),
                    (action,),
                    ({"name": "search", "arguments": {}},),
                )
            )


@pytest.mark.parametrize("stage", ["parse", "compile", "wrap", "canonicalize"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_binding_check_in_finally_overrides_payload_errors_after_rebinding(
    stage: str,
    error_type: type[Exception],
) -> None:
    source = _RebindThenRaiseSourceAdapter(_base_variant(), stage, error_type)
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    action = SemanticAction("search", {})
    with pytest.raises(TransformValidationError, match="binding") as caught:
        if stage == "parse":
            adapter.surface_to_semantic({"name": "lookup", "arguments": {}})
        elif stage == "compile":
            adapter.semantic_to_base_calls(action)
        elif stage == "wrap":
            adapter.base_observation_to_surface(
                {"name": "lookup", "arguments": {}},
                (action,),
                (({"ok": True},),),
            )
        else:
            adapter.canonicalize_trace(
                ExecutionTrace(
                    ({"name": "lookup", "arguments": {}},),
                    (action,),
                    ({"name": "search", "arguments": {}},),
                )
            )
    assert "PRIVATE_EXCEPTION_PAYLOAD" not in str(caught.value)


def test_rename_adapter_rejects_preexisting_source_final_and_transform_tamper() -> None:
    base = _base_variant()
    source = _RecordingIdentityAdapter(base)
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    replacement = _base_variant(reverse_manifest=True)
    object.__setattr__(source, "_variant", replacement)
    with pytest.raises(TransformValidationError, match="binding"):
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})

    source = _RecordingIdentityAdapter(base)
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    object.__setattr__(adapter, "_variant", base)
    with pytest.raises(TransformValidationError, match="binding"):
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})

    source = _RecordingIdentityAdapter(base)
    adapter = apply_rename(
        source,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    object.__setattr__(adapter.transform, "canonical_to_surface", {"search": "changed"})
    with pytest.raises(TransformValidationError, match="integrity"):
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})


@pytest.mark.parametrize(
    "field_name",
    [
        "source_variant",
        "variant",
        "operator",
        "_surface_to_canonical",
        "_canonical_snapshot",
        "_snapshot_fingerprint",
    ],
)
def test_rename_adapter_rejects_transform_component_tamper(field_name: str) -> None:
    base = _base_variant()
    adapter = apply_rename(
        _RecordingIdentityAdapter(base),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    if field_name == "operator":
        replacement: object = _entry(base)
    elif field_name == "_surface_to_canonical":
        replacement = {"lookup": "changed"}
    elif field_name == "_canonical_snapshot":
        replacement = b"changed"
    elif field_name == "_snapshot_fingerprint":
        replacement = "b" * 64
    else:
        replacement = _base_variant(reverse_manifest=True)
    object.__setattr__(adapter.transform, field_name, replacement)
    with pytest.raises(TransformValidationError, match="integrity"):
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})


def test_rename_adapter_rejects_same_object_source_variant_content_tamper() -> None:
    base = _base_variant()
    adapter = apply_rename(
        _RecordingIdentityAdapter(base),
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    replacement = SchemaVariant(
        base.variant_id,
        base.tools,
        {"operator": "mutated", "seed": 3},
    )
    object.__setattr__(base, "manifest", replacement.manifest)
    object.__setattr__(base, "_manifest_canonical", replacement._manifest_canonical)
    with pytest.raises(TransformValidationError, match=r"integrity|binding"):
        adapter.surface_to_semantic({"name": "lookup", "arguments": {}})


def _tamper_frozen_mapping_entry(
    mapping: Mapping[str, JSONValue],
    key: str,
    replacement: JSONValue,
) -> None:
    items = object.__getattribute__(mapping, "_items")
    changed = tuple(
        (entry_key, replacement if entry_key == key else value) for entry_key, value in items
    )
    object.__setattr__(mapping, "_items", changed)
    object.__setattr__(
        mapping,
        "_index",
        {entry_key: index for index, (entry_key, _) in enumerate(changed)},
    )


def test_construction_boundary_deep_validates_nested_frozen_schema_graph() -> None:
    # Online guards intentionally attest only immutable roots in O(1).  Full graph
    # validation belongs here and in contract/Gate admission; code capable of using
    # object.__setattr__ on private frozen nodes could also rewrite an online seal.
    base = _base_variant()
    properties = base.tools[0].input_schema["properties"]
    assert isinstance(properties, Mapping)
    value_schema = properties["value"]
    assert isinstance(value_schema, Mapping)
    _tamper_frozen_mapping_entry(value_schema, "type", "integer")

    with pytest.raises(TransformValidationError, match="base_variant"):
        build_rename_transform(
            base,
            tool_name_mapping={"search": "lookup"},
            seed=7,
        )


class _LyingPublicVariantAdapter(_RecordingIdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._public_variant = SchemaVariant(
            "public-lie-v1",
            variant.tools,
            {"operator": "public_lie", "seed": 4},
        )

    @property
    def variant(self) -> SchemaVariant:
        return self._public_variant


def test_rename_adapter_rejects_public_variant_lie_even_when_raw_binding_is_correct() -> None:
    with pytest.raises(TransformValidationError, match="binding"):
        apply_rename(
            _LyingPublicVariantAdapter(_base_variant()),
            tool_name_mapping={"search": "lookup"},
            seed=7,
        )


def test_apply_rename_requires_a_source_adapter_bound_to_the_exact_base_variant() -> None:
    first = _base_variant()
    equal_but_distinct = _base_variant()
    source = _RecordingIdentityAdapter(equal_but_distinct)
    transform = build_rename_transform(
        first,
        tool_name_mapping={"search": "lookup"},
        seed=7,
    )
    with pytest.raises(TransformValidationError, match="binding"):
        RenameAdapter(source, transform)
