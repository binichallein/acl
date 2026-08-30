"""Immutable canonical types and stable manifest serialization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import ItemsView, Iterator, Mapping, ValuesView
from dataclasses import dataclass, field
from typing import TypeAlias, cast

JSONScalar: TypeAlias = bool | int | float | str | None
JSONValue: TypeAlias = (
    JSONScalar | Mapping[str, "JSONValue"] | list["JSONValue"] | tuple["JSONValue", ...]
)
_MAX_SAFE_INTEGER = 2**53 - 1
_MAX_JSON_CONTAINER_DEPTH = 128


class _FrozenJSONMapping(Mapping[str, JSONValue]):
    """Private immutable mapping backed only by an exact tuple graph."""

    __slots__ = ("_index", "_items")

    def __init__(self, items: tuple[tuple[str, JSONValue], ...]) -> None:
        if type(items) is not tuple:
            raise ValueError("frozen mapping state is invalid")
        object.__setattr__(self, "_items", items)
        index: dict[str, int] = {}
        for position, entry in enumerate(items):
            if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str:
                raise ValueError("frozen mapping state is invalid")
            key = entry[0]
            if key in index:
                raise ValueError("frozen mapping keys must be unique")
            index[key] = position
        object.__setattr__(self, "_index", index)

    def __getattribute__(self, name: str) -> object:
        if name == "_index":
            raise AttributeError("frozen mapping lookup index is private")
        return object.__getattribute__(self, name)

    def _raw_items(self) -> tuple[tuple[str, JSONValue], ...]:
        items = object.__getattribute__(self, "_items")
        if type(items) is not tuple:
            raise ValueError("frozen mapping state is invalid")
        for entry in items:
            if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str:
                raise ValueError("frozen mapping state is invalid")
        return cast(tuple[tuple[str, JSONValue], ...], items)

    def __getitem__(self, key: str) -> JSONValue:
        items = object.__getattribute__(self, "_items")
        index = object.__getattribute__(self, "_index")
        if type(items) is not tuple or type(index) is not dict:
            raise ValueError("frozen mapping state is invalid")
        position = dict.get(index, key)
        if position is None:
            raise KeyError(key)
        if type(position) is not int or position < 0 or position >= len(items):
            raise ValueError("frozen mapping state is invalid")
        entry = items[position]
        if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str:
            raise ValueError("frozen mapping state is invalid")
        if entry[0] != key:
            raise ValueError("frozen mapping state is invalid")
        return cast(JSONValue, entry[1])

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._raw_items())

    def __len__(self) -> int:
        items = object.__getattribute__(self, "_items")
        if type(items) is not tuple:
            raise ValueError("frozen mapping state is invalid")
        return len(items)

    def items(self) -> ItemsView[str, JSONValue]:
        """Return the immutable ordered entries without repeated key lookup."""

        return _FrozenJSONItemsView(self)

    def values(self) -> ValuesView[JSONValue]:
        """Return values in insertion order without repeated key lookup."""

        return _FrozenJSONValuesView(self)

    def __repr__(self) -> str:
        rendered = ", ".join(f"{key!r}: {value!r}" for key, value in self._raw_items())
        return f"FrozenJSONMapping({{{rendered}}})"

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("frozen JSON mappings are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("frozen JSON mappings are immutable")


class _FrozenJSONItemsView(ItemsView[str, JSONValue]):
    __slots__ = ()

    def __iter__(self) -> Iterator[tuple[str, JSONValue]]:
        mapping = cast(_FrozenJSONMapping, object.__getattribute__(self, "_mapping"))
        return iter(mapping._raw_items())


class _FrozenJSONValuesView(ValuesView[JSONValue]):
    __slots__ = ()

    def __iter__(self) -> Iterator[JSONValue]:
        mapping = cast(_FrozenJSONMapping, object.__getattribute__(self, "_mapping"))
        return (value for _, value in mapping._raw_items())


def _frozen_mapping_index_matches(mapping: _FrozenJSONMapping) -> bool:
    """Validate the private index without invoking injected key behavior."""

    try:
        items = object.__getattribute__(mapping, "_items")
        index = object.__getattribute__(mapping, "_index")
    except AttributeError:
        return False
    if type(items) is not tuple or type(index) is not dict:
        return False
    index_entries = tuple(dict.items(index))
    if len(index_entries) != len(items):
        return False
    for position, (item_entry, index_entry) in enumerate(
        zip(items, index_entries, strict=True)
    ):
        if type(item_entry) is not tuple or len(item_entry) != 2:
            return False
        if type(item_entry[0]) is not str:
            return False
        index_key, index_position = index_entry
        if type(index_key) is not str or type(index_position) is not int:
            return False
        if index_key != item_entry[0] or index_position != position:
            return False
    return True


def _require_utf8_text(value: str, context: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{context} must be valid UTF-8 text") from error
    return value


def _require_non_blank_string(value: object, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{context} must be a non-blank string")
    return _require_utf8_text(value, context)


def _freeze_json(
    value: object,
    context: str,
    active: set[int],
    container_depth: int,
    *,
    allow_wide_integers: bool,
) -> JSONValue:
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        return _require_utf8_text(value, context)
    if type(value) is int:
        if (
            not allow_wide_integers
            and not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER
        ):
            raise ValueError(
                f"{context} must be an integer in the I-JSON safe range "
                f"[-{_MAX_SAFE_INTEGER}, {_MAX_SAFE_INTEGER}]"
            )
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} must be a finite float")
        return value

    if type(value) is _FrozenJSONMapping:
        if container_depth >= _MAX_JSON_CONTAINER_DEPTH:
            raise ValueError(
                f"{context} exceeds maximum JSON container depth {_MAX_JSON_CONTAINER_DEPTH}"
            )
        identity = id(value)
        if identity in active:
            raise ValueError(f"{context} contains a cycle")
        active.add(identity)
        try:
            if not _frozen_mapping_index_matches(value):
                raise ValueError(f"{context} contains invalid frozen mapping state")
            entries = cast(
                tuple[tuple[str, JSONValue], ...],
                object.__getattribute__(value, "_items"),
            )
            return _FrozenJSONMapping(
                tuple(
                    (
                        key,
                        _freeze_json(
                            item,
                            f"{context}.{key}",
                            active,
                            container_depth + 1,
                            allow_wide_integers=allow_wide_integers,
                        ),
                    )
                    for key, item in entries
                )
            )
        finally:
            active.remove(identity)

    is_mapping = isinstance(value, Mapping)
    is_array = isinstance(value, (list, tuple))
    if (is_mapping or is_array) and container_depth >= _MAX_JSON_CONTAINER_DEPTH:
        raise ValueError(
            f"{context} exceeds maximum JSON container depth {_MAX_JSON_CONTAINER_DEPTH}"
        )

    if is_mapping:
        identity = id(value)
        if identity in active:
            raise ValueError(f"{context} contains a cycle")
        active.add(identity)
        try:
            frozen: dict[str, JSONValue] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(f"{context} has non-string key: {key!r}")
                _require_utf8_text(key, f"{context} mapping key {key!r}")
                frozen[key] = _freeze_json(
                    item,
                    f"{context}.{key}",
                    active,
                    container_depth + 1,
                    allow_wide_integers=allow_wide_integers,
                )
            return _FrozenJSONMapping(tuple(frozen.items()))
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
                    allow_wide_integers=allow_wide_integers,
                )
                for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)

    raise ValueError(f"{context} contains unsupported JSON value of type {type(value).__name__}")


def _freeze_json_root(value: object, context: str) -> JSONValue:
    try:
        return _freeze_json(
            value,
            context,
            set(),
            0,
            allow_wide_integers=False,
        )
    except RecursionError as error:
        raise ValueError(
            f"{context} exceeds maximum JSON container depth {_MAX_JSON_CONTAINER_DEPTH}"
        ) from error


def _freeze_schema_json_root(value: object, context: str) -> JSONValue:
    """Freeze schema-document JSON while preserving exact Python integers."""

    try:
        return _freeze_json(
            value,
            context,
            set(),
            0,
            allow_wide_integers=True,
        )
    except RecursionError as error:
        raise ValueError(
            f"{context} exceeds maximum JSON container depth {_MAX_JSON_CONTAINER_DEPTH}"
        ) from error


def _freeze_mapping(value: object, context: str) -> Mapping[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return cast(Mapping[str, JSONValue], _freeze_json_root(value, context))


def _freeze_schema_mapping(value: object, context: str) -> Mapping[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return cast(Mapping[str, JSONValue], _freeze_schema_json_root(value, context))


def _is_frozen_json_domain(value: object, *, allow_wide_integers: bool) -> bool:
    def visit(item: object, active: set[int], container_depth: int) -> bool:
        if item is None or type(item) is bool:
            return True
        if type(item) is str:
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                return False
            return True
        if type(item) is int:
            return allow_wide_integers or -_MAX_SAFE_INTEGER <= item <= _MAX_SAFE_INTEGER
        if type(item) is float:
            return math.isfinite(item)
        if type(item) not in (tuple, _FrozenJSONMapping):
            return False
        if container_depth >= _MAX_JSON_CONTAINER_DEPTH:
            return False
        identity = id(item)
        if identity in active:
            return False
        active.add(identity)
        try:
            if type(item) is tuple:
                return all(visit(child, active, container_depth + 1) for child in item)
            if not _frozen_mapping_index_matches(item):
                return False
            try:
                entries = object.__getattribute__(item, "_items")
            except AttributeError:
                return False
            if type(entries) is not tuple:
                return False
            keys: set[str] = set()
            for entry in entries:
                if type(entry) is not tuple or len(entry) != 2:
                    return False
                key = entry[0]
                if type(key) is not str or key in keys:
                    return False
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError:
                    return False
                keys.add(key)
                if not visit(entry[1], active, container_depth + 1):
                    return False
            return True
        finally:
            active.remove(identity)

    return visit(value, set(), 0)


def _is_frozen_json(value: object) -> bool:
    return _is_frozen_json_domain(value, allow_wide_integers=False)


def _is_frozen_schema_json(value: object) -> bool:
    return _is_frozen_json_domain(value, allow_wide_integers=True)


def _is_frozen_mapping(value: object) -> bool:
    return type(value) is _FrozenJSONMapping and _is_frozen_json(value)


def _is_frozen_schema_mapping(value: object) -> bool:
    return type(value) is _FrozenJSONMapping and _is_frozen_schema_json(value)


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
    if type(value) is _FrozenJSONMapping:
        if not _frozen_mapping_index_matches(value):
            raise ValueError("frozen mapping state is invalid")
        entries = cast(
            tuple[tuple[str, JSONValue], ...],
            object.__getattribute__(value, "_items"),
        )
        return {key: _to_json_container(item) for key, item in entries}
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


def _schema_canonical_json_bytes_from_frozen(
    value: JSONValue,
    context: str,
) -> bytes:
    if not _is_frozen_schema_json(value):
        raise ValueError(f"{context} contains invalid frozen schema JSON")
    try:
        return _schema_json_text_from_frozen(value).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise ValueError(
            f"{context} cannot be serialized as canonical schema JSON: {error}"
        ) from error


def _schema_integer_text(value: int) -> str:
    if value == 0:
        return "0"
    negative = value < 0
    magnitude = -value if negative else value
    chunks: list[int] = []
    while magnitude:
        magnitude, chunk = divmod(magnitude, 1_000_000_000)
        chunks.append(chunk)
    most_significant = chunks.pop()
    text = f"{most_significant:d}" + "".join(
        f"{chunk:09d}" for chunk in reversed(chunks)
    )
    return f"-{text}" if negative else text


def _schema_json_text_from_frozen(value: JSONValue) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is str:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if type(value) is int:
        return _schema_integer_text(value)
    if type(value) is float:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if type(value) is tuple:
        return "[" + ",".join(_schema_json_text_from_frozen(item) for item in value) + "]"
    if type(value) is _FrozenJSONMapping:
        entries = cast(
            tuple[tuple[str, JSONValue], ...],
            object.__getattribute__(value, "_items"),
        )
        return "{" + ",".join(
            json.dumps(key, ensure_ascii=False, allow_nan=False)
            + ":"
            + _schema_json_text_from_frozen(item)
            for key, item in sorted(entries, key=lambda entry: entry[0])
        ) + "}"
    raise ValueError("schema JSON contains an unsupported frozen value")


def _schema_canonical_json_bytes(value: JSONValue) -> bytes:
    """Serialize schema-document JSON without narrowing exact Python integers."""

    frozen = _freeze_schema_json_root(value, "schema")
    return _schema_canonical_json_bytes_from_frozen(frozen, "schema")


def _schema_json_container(
    value: JSONValue,
) -> JSONScalar | dict[str, object] | list[object]:
    """Return detached built-in containers for validated schema-document JSON."""

    frozen = _freeze_schema_json_root(value, "schema")
    return _to_json_container(frozen)


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
        return self.name == other.name and self._arguments_canonical == other._arguments_canonical


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
        if type(self.description) is not str:
            raise ValueError("description must be a string")
        _require_utf8_text(self.description, "description")
        input_schema = _freeze_schema_mapping(self.input_schema, "input_schema")
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(
            self,
            "_input_schema_canonical",
            _schema_canonical_json_bytes_from_frozen(input_schema, "input_schema"),
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


def _semantic_action_has_canonical_shape(value: object) -> bool:
    if type(value) is not SemanticAction:
        return False
    action = cast(SemanticAction, value)
    return (
        type(object.__getattribute__(action, "name")) is str
        and _is_frozen_mapping(object.__getattribute__(action, "arguments"))
        and type(object.__getattribute__(action, "_arguments_canonical")) is bytes
    )


def _surface_tool_spec_has_canonical_shape(value: object) -> bool:
    if type(value) is not SurfaceToolSpec:
        return False
    tool = cast(SurfaceToolSpec, value)
    return (
        type(object.__getattribute__(tool, "name")) is str
        and type(object.__getattribute__(tool, "description")) is str
        and _is_frozen_schema_mapping(object.__getattribute__(tool, "input_schema"))
        and type(object.__getattribute__(tool, "_input_schema_canonical")) is bytes
    )


def _schema_variant_has_canonical_shape(value: object) -> bool:
    if type(value) is not SchemaVariant:
        return False
    variant = cast(SchemaVariant, value)
    tools = object.__getattribute__(variant, "tools")
    return (
        type(object.__getattribute__(variant, "variant_id")) is str
        and type(tools) is tuple
        and all(_surface_tool_spec_has_canonical_shape(tool) for tool in tools)
        and _is_frozen_mapping(object.__getattribute__(variant, "manifest"))
        and type(object.__getattribute__(variant, "_manifest_canonical")) is bytes
    )


def _execution_trace_has_canonical_shape(value: object) -> bool:
    if type(value) is not ExecutionTrace:
        return False
    trace = cast(ExecutionTrace, value)
    surface_calls = object.__getattribute__(trace, "surface_calls")
    semantic_actions = object.__getattribute__(trace, "semantic_actions")
    base_calls = object.__getattribute__(trace, "base_calls")
    return (
        type(surface_calls) is tuple
        and all(_is_frozen_mapping(call) for call in surface_calls)
        and type(semantic_actions) is tuple
        and all(_semantic_action_has_canonical_shape(action) for action in semantic_actions)
        and type(base_calls) is tuple
        and all(_is_frozen_mapping(call) for call in base_calls)
        and type(object.__getattribute__(trace, "_surface_calls_canonical")) is bytes
        and type(object.__getattribute__(trace, "_base_calls_canonical")) is bytes
    )
