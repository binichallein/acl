"""Immutable canonical types and stable manifest serialization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias, cast

JSONScalar: TypeAlias = bool | int | float | str | None
JSONValue: TypeAlias = (
    JSONScalar
    | Mapping[str, "JSONValue"]
    | list["JSONValue"]
    | tuple["JSONValue", ...]
)
_MAX_SAFE_INTEGER = 2**53 - 1
_MAX_JSON_CONTAINER_DEPTH = 128


def _require_utf8_text(value: str, context: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must be valid UTF-8 text") from error
    return value


def _require_non_blank_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-blank string")
    return _require_utf8_text(value, context)


def _freeze_json(
    value: object,
    context: str,
    active: set[int],
    container_depth: int,
) -> JSONValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _require_utf8_text(value, context)
    if isinstance(value, int):
        if not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER:
            raise ValueError(
                f"{context} must be an integer in the I-JSON safe range "
                f"[-{_MAX_SAFE_INTEGER}, {_MAX_SAFE_INTEGER}]"
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} must be a finite float")
        return value

    is_mapping = isinstance(value, Mapping)
    is_array = isinstance(value, (list, tuple))
    if (is_mapping or is_array) and container_depth >= _MAX_JSON_CONTAINER_DEPTH:
        raise ValueError(
            f"{context} exceeds maximum JSON container depth "
            f"{_MAX_JSON_CONTAINER_DEPTH}"
        )

    if is_mapping:
        identity = id(value)
        if identity in active:
            raise ValueError(f"{context} contains a cycle")
        active.add(identity)
        try:
            frozen: dict[str, JSONValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{context} has non-string key: {key!r}")
                _require_utf8_text(key, f"{context} mapping key {key!r}")
                frozen[key] = _freeze_json(
                    item,
                    f"{context}.{key}",
                    active,
                    container_depth + 1,
                )
            return MappingProxyType(frozen)
        finally:
            active.remove(identity)

    if is_array:
        identity = id(value)
        if identity in active:
            raise ValueError(f"{context} contains a cycle")
        active.add(identity)
        try:
            return tuple(
                _freeze_json(
                    item,
                    f"{context}[{index}]",
                    active,
                    container_depth + 1,
                )
                for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)

    raise ValueError(
        f"{context} contains unsupported JSON value of type {type(value).__name__}"
    )


def _freeze_json_root(value: object, context: str) -> JSONValue:
    try:
        return _freeze_json(value, context, set(), 0)
    except RecursionError as error:
        raise ValueError(
            f"{context} exceeds maximum JSON container depth "
            f"{_MAX_JSON_CONTAINER_DEPTH}"
        ) from error


def _freeze_mapping(value: object, context: str) -> Mapping[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return cast(Mapping[str, JSONValue], _freeze_json_root(value, context))


def _freeze_call_sequence(
    value: object,
    context: str,
) -> tuple[Mapping[str, JSONValue], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list or tuple")
    frozen = _freeze_json_root(value, context)
    if not isinstance(frozen, tuple) or any(not isinstance(call, Mapping) for call in frozen):
        raise ValueError(f"{context} must contain only mappings")
    return cast(tuple[Mapping[str, JSONValue], ...], frozen)


def _to_json_container(value: JSONValue) -> JSONScalar | dict[str, object] | list[object]:
    if isinstance(value, Mapping):
        return {key: _to_json_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_container(item) for item in value]
    return value


def _canonical_json_bytes_from_frozen(value: JSONValue, context: str) -> bytes:
    try:
        serializable = _to_json_container(value)
        text = json.dumps(
            serializable,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise ValueError(f"{context} cannot be serialized as canonical JSON: {error}") from error


def canonical_json_bytes(value: JSONValue) -> bytes:
    """Serialize JSON to UTF-8 with a fixed 128-container nesting limit."""

    frozen = _freeze_json_root(value, "value")
    return _canonical_json_bytes_from_frozen(frozen, "value")


def manifest_sha256(manifest: JSONValue) -> str:
    """Return a stable lowercase SHA256 digest for a parsed JSON manifest."""

    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class SemanticAction:
    """A tool action expressed in canonical task semantics."""

    name: str
    arguments: Mapping[str, JSONValue]
    _arguments_canonical: bytes = field(init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_blank_string(self.name, "name"))
        arguments = _freeze_mapping(self.arguments, "arguments")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(
            self,
            "_arguments_canonical",
            _canonical_json_bytes_from_frozen(arguments, "arguments"),
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.name == other.name
            and self._arguments_canonical == other._arguments_canonical
        )


@dataclass(frozen=True, slots=True, eq=False)
class SurfaceToolSpec:
    """One tool definition presented to an agent under a schema variant."""

    name: str
    description: str
    input_schema: Mapping[str, JSONValue]
    _input_schema_canonical: bytes = field(init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_non_blank_string(self.name, "name"))
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        _require_utf8_text(self.description, "description")
        input_schema = _freeze_mapping(self.input_schema, "input_schema")
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(
            self,
            "_input_schema_canonical",
            _canonical_json_bytes_from_frozen(input_schema, "input_schema"),
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.name == other.name
            and self.description == other.description
            and self._input_schema_canonical == other._input_schema_canonical
        )


@dataclass(frozen=True, slots=True, eq=False)
class SchemaVariant:
    """An immutable surface schema and the manifest that generated it."""

    variant_id: str
    tools: tuple[SurfaceToolSpec, ...]
    manifest: Mapping[str, JSONValue]
    _manifest_canonical: bytes = field(init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variant_id",
            _require_non_blank_string(self.variant_id, "variant_id"),
        )
        if not isinstance(self.tools, (list, tuple)):
            raise ValueError("tools must be a list or tuple")
        tools = tuple(self.tools)
        if not tools:
            raise ValueError("tools must contain at least one tool")
        if any(not isinstance(tool, SurfaceToolSpec) for tool in tools):
            raise ValueError("tools must contain only SurfaceToolSpec values")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        object.__setattr__(self, "tools", tools)
        manifest = _freeze_mapping(self.manifest, "manifest")
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(
            self,
            "_manifest_canonical",
            _canonical_json_bytes_from_frozen(manifest, "manifest"),
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.variant_id == other.variant_id
            and self.tools == other.tools
            and self._manifest_canonical == other._manifest_canonical
        )


@dataclass(frozen=True, slots=True, eq=False)
class ExecutionTrace:
    """Ordered surface, semantic, and base channels for an execution."""

    surface_calls: tuple[Mapping[str, JSONValue], ...]
    semantic_actions: tuple[SemanticAction, ...]
    base_calls: tuple[Mapping[str, JSONValue], ...]
    _surface_calls_canonical: bytes = field(init=False, repr=False)
    _base_calls_canonical: bytes = field(init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        surface_calls = _freeze_call_sequence(self.surface_calls, "surface_calls")
        if not isinstance(self.semantic_actions, (list, tuple)):
            raise ValueError("semantic_actions must be a list or tuple")
        semantic_actions = tuple(self.semantic_actions)
        if any(not isinstance(action, SemanticAction) for action in semantic_actions):
            raise ValueError("semantic_actions must contain only SemanticAction values")
        base_calls = _freeze_call_sequence(self.base_calls, "base_calls")
        object.__setattr__(self, "surface_calls", surface_calls)
        object.__setattr__(self, "semantic_actions", semantic_actions)
        object.__setattr__(self, "base_calls", base_calls)
        object.__setattr__(
            self,
            "_surface_calls_canonical",
            _canonical_json_bytes_from_frozen(surface_calls, "surface_calls"),
        )
        object.__setattr__(
            self,
            "_base_calls_canonical",
            _canonical_json_bytes_from_frozen(base_calls, "base_calls"),
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (
            self._surface_calls_canonical == other._surface_calls_canonical
            and self.semantic_actions == other.semantic_actions
            and self._base_calls_canonical == other._base_calls_canonical
        )
