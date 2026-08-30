"""Pure catalog snapshots for the external AppWorld runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from jsonschema import Draft202012Validator, FormatChecker

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    _freeze_json_root,
    canonical_json_bytes,
)

_CATALOG_ERROR = "AppWorld catalog is invalid"
_SCHEMA_ERROR = "AppWorld schema is invalid"
_SEMANTIC_UNAVAILABLE = "AppWorld semantic behavior is unavailable"
_OUTER_KEYS = frozenset({"type", "function"})
_FUNCTION_KEYS = frozenset({"name", "description", "parameters"})
_REQUESTER_CONTROL_NAMES = frozenset(
    {
        "_api_name",
        "_app_name",
        "_system_datetime",
        "client",
        "raise_on_failure",
        "show",
        "track",
    }
)
_VARIANT_ID = "appworld-source-v1"
_VARIANT_MANIFEST: Mapping[str, JSONValue] = {
    "kind": "appworld_source_interface",
    "schema_policy": "closed-root-v1",
}


class _CatalogSnapshotError(ValueError):
    """Internal marker converted to one payload-free public error."""


class _SchemaSnapshotError(ValueError):
    """Internal marker converted to one payload-free public error."""


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _ToolBinding:
    app_name: str
    api_name: str
    schema: Mapping[str, JSONValue]
    schema_canonical: bytes
    validator: Draft202012Validator


def _snapshot_json(value: object, active: set[int]) -> object:
    if value is None or type(value) in (bool, str, int, float):
        return value

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise _CatalogSnapshotError
        active.add(identity)
        try:
            raw_entries = tuple(value.items())
            entries = tuple(tuple(entry) for entry in raw_entries)
            snapshot: dict[str, object] = {}
            for entry in entries:
                if len(entry) != 2 or type(entry[0]) is not str:
                    raise _CatalogSnapshotError
                key = entry[0]
                if key in snapshot:
                    raise _CatalogSnapshotError
                snapshot[key] = _snapshot_json(entry[1], active)
            return snapshot
        except _CatalogSnapshotError:
            raise
        except Exception:
            raise _CatalogSnapshotError from None
        finally:
            active.remove(identity)

    if type(value) in (list, tuple):
        identity = id(value)
        if identity in active:
            raise _CatalogSnapshotError
        active.add(identity)
        try:
            items = tuple(_snapshot_json(item, active) for item in value)
            return list(items) if type(value) is list else items
        except _CatalogSnapshotError:
            raise
        except Exception:
            raise _CatalogSnapshotError from None
        finally:
            active.remove(identity)

    raise _CatalogSnapshotError


def _snapshot_catalog(
    function_catalog: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], JSONValue, bytes]:
    if isinstance(function_catalog, (str, bytes, bytearray)) or not isinstance(
        function_catalog, Sequence
    ):
        raise ValueError(_CATALOG_ERROR)
    try:
        raw_entries = tuple(function_catalog)
        if not raw_entries:
            raise _CatalogSnapshotError
        snapshots = tuple(_snapshot_json(entry, set()) for entry in raw_entries)
        if any(type(entry) is not dict for entry in snapshots):
            raise _CatalogSnapshotError
        plain = cast(tuple[dict[str, object], ...], snapshots)
        frozen = _freeze_json_root(plain, "AppWorld catalog")
        canonical = canonical_json_bytes(frozen)
        return plain, frozen, canonical
    except _CatalogSnapshotError:
        raise ValueError(_CATALOG_ERROR) from None
    except Exception:
        raise ValueError(_CATALOG_ERROR) from None


def _split_native_name(value: object) -> tuple[str, str]:
    if type(value) is not str:
        raise _CatalogSnapshotError
    app_name, separator, api_name = value.partition("__")
    if (
        separator != "__"
        or not app_name
        or not api_name
        or not app_name.isascii()
        or not api_name.isascii()
        or not app_name.isidentifier()
        or not api_name.isidentifier()
    ):
        raise _CatalogSnapshotError
    return app_name, api_name


def _closed_schema(value: object) -> dict[str, object]:
    try:
        if type(value) is not dict:
            raise _SchemaSnapshotError
        schema = dict(value)
        if schema.get("type") != "object":
            raise _SchemaSnapshotError
        properties = schema.get("properties")
        if type(properties) is not dict:
            raise _SchemaSnapshotError
        if "patternProperties" in schema:
            raise _SchemaSnapshotError
        if _REQUESTER_CONTROL_NAMES.intersection(properties):
            raise _SchemaSnapshotError
        required = schema.get("required")
        if type(required) is not list:
            raise _SchemaSnapshotError
        if (
            any(type(name) is not str for name in required)
            or len(required) != len(set(required))
            or any(name not in properties for name in required)
        ):
            raise _SchemaSnapshotError
        if "additionalProperties" in schema:
            if schema["additionalProperties"] is not False:
                raise _SchemaSnapshotError
        else:
            schema["additionalProperties"] = False
        Draft202012Validator.check_schema(schema)
        return schema
    except _SchemaSnapshotError:
        raise ValueError(_SCHEMA_ERROR) from None
    except Exception:
        raise ValueError(_SCHEMA_ERROR) from None


def _catalog_records(
    entries: tuple[dict[str, object], ...],
) -> tuple[tuple[str, str, str, str, dict[str, object]], ...]:
    records: list[tuple[str, str, str, str, dict[str, object]]] = []
    names: set[str] = set()
    try:
        for entry in entries:
            if set(entry) != _OUTER_KEYS or entry.get("type") != "function":
                raise _CatalogSnapshotError
            function = entry.get("function")
            if type(function) is not dict or set(function) != _FUNCTION_KEYS:
                raise _CatalogSnapshotError
            name = function.get("name")
            app_name, api_name = _split_native_name(name)
            if cast(str, name) in names:
                raise _CatalogSnapshotError
            names.add(cast(str, name))
            description = function.get("description")
            if type(description) is not str:
                raise _CatalogSnapshotError
            description.encode("utf-8")
            schema = _closed_schema(function.get("parameters"))
            records.append((cast(str, name), app_name, api_name, description, schema))
        return tuple(sorted(records, key=lambda record: record[0]))
    except _CatalogSnapshotError:
        raise ValueError(_CATALOG_ERROR) from None
    except UnicodeEncodeError:
        raise ValueError(_CATALOG_ERROR) from None


class AppWorldSemanticAdapter(SemanticAdapter):
    """Immutable snapshot of one world's full non-admin AppWorld catalog."""

    __slots__ = (
        "_bindings",
        "_bindings_seal",
        "_catalog_canonical",
        "_catalog_snapshot",
        "_validators",
        "_validators_seal",
        "_variant_seal",
    )

    def __init__(
        self,
        function_catalog: Sequence[Mapping[str, object]],
    ) -> None:
        entries, catalog_snapshot, catalog_canonical = _snapshot_catalog(function_catalog)
        records = _catalog_records(entries)

        tools: list[SurfaceToolSpec] = []
        bindings: dict[str, _ToolBinding] = {}
        validators: dict[str, Draft202012Validator] = {}
        try:
            for name, app_name, api_name, description, schema in records:
                tool = SurfaceToolSpec(name, description, schema)
                validator = Draft202012Validator(
                    tool.input_schema,
                    format_checker=FormatChecker(),
                )
                tools.append(tool)
                validators[name] = validator
                bindings[name] = _ToolBinding(
                    app_name,
                    api_name,
                    tool.input_schema,
                    canonical_json_bytes(tool.input_schema),
                    validator,
                )
            variant = SchemaVariant(_VARIANT_ID, tuple(tools), _VARIANT_MANIFEST)
        except Exception:
            raise ValueError(_CATALOG_ERROR) from None

        super().__init__(variant)
        validator_root = MappingProxyType(validators)
        binding_root = MappingProxyType(bindings)
        self._variant_seal = variant
        self._catalog_snapshot = catalog_snapshot
        self._catalog_canonical = catalog_canonical
        self._validators = validator_root
        self._validators_seal = validator_root
        self._bindings = binding_root
        self._bindings_seal = binding_root

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        raise ValueError(_SEMANTIC_UNAVAILABLE)

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        raise ValueError(_SEMANTIC_UNAVAILABLE)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        raise ValueError(_SEMANTIC_UNAVAILABLE)

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        raise ValueError(_SEMANTIC_UNAVAILABLE)


def build_appworld_adapter(
    function_catalog: Sequence[Mapping[str, object]],
) -> AppWorldSemanticAdapter:
    """Build one pure adapter from a detached AppWorld function catalog snapshot."""

    return AppWorldSemanticAdapter(function_catalog)


__all__ = ["AppWorldSemanticAdapter", "build_appworld_adapter"]
