"""Isolated, in-process execution of validated AppWorld native calls.

The module deliberately has no import-time dependency on AppWorld.  A caller
owns the lifecycle of the injected live world; this executor is one-shot and
never closes, resets, or rolls that world back.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import TypeVar, cast

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.benchmarks.appworld_replay import (
    canonical_state_sha256,
    evaluator_sha256,
)
from toolshift.contracts.trace import PhysicalCallEffect, base_call_fingerprint
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SemanticAction,
    _execution_trace_has_canonical_shape,
    _freeze_json_root,
    _freeze_mapping,
    _semantic_action_has_canonical_shape,
    canonical_json_bytes,
)

_CONFIGURATION_ERROR = "AppWorld executor configuration is invalid"
_EXECUTION_ERROR = "AppWorld episode execution failed"
_INTEGRITY_ERROR = "AppWorld executor integrity validation failed"
_CALL_KEYS = frozenset({"name", "arguments"})
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
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_EFFECT_DOMAIN = b"toolshift.appworld.effect.v1\0"
_TRANSITION_DOMAIN = b"toolshift.appworld.transition.v1\0"
_INTEGRITY_TOKEN = object()
_ENTRY_LATCH_TOKEN = object()
_TUPLE_ITERATOR_TYPE = type(iter(()))
_READY = "ready"
_RUNNING = "running"
_CONSUMED = "consumed"
_ResultT = TypeVar("_ResultT")


class _ExecutorIntegrityError(ValueError):
    """Internal marker converted to one payload-free public error."""


@dataclass(frozen=True, slots=True, repr=False)
class _CallableSeal:
    callback: Callable[..., object]
    bound_self: object | None
    bound_function: object | None


@dataclass(frozen=True, slots=True, repr=False)
class _ExecutedStep:
    surface_call: Mapping[str, JSONValue]
    action: SemanticAction
    base_call: Mapping[str, JSONValue]
    base_observation: JSONValue
    surface_observation: JSONValue
    before_state_sha256: str
    after_state_sha256: str


@dataclass(frozen=True, slots=True, repr=False)
class _EpisodeRecord:
    steps: tuple[_ExecutedStep, ...]
    trace: ExecutionTrace
    initial_state_sha256: str
    final_state_sha256: str
    evaluator_score: JSONValue
    evaluator_digest: str
    effects: tuple[PhysicalCallEffect, ...]
    transition_digest: str


def _make_callable_seal(value: object) -> _CallableSeal:
    if not callable(value):
        raise ValueError(_CONFIGURATION_ERROR)
    callback = cast(Callable[..., object], value)
    if isinstance(callback, MethodType):
        return _CallableSeal(callback, callback.__self__, callback.__func__)
    return _CallableSeal(callback, None, None)


def _callable_matches(value: object, seal: _CallableSeal) -> bool:
    if not callable(value):
        return False
    if seal.bound_self is None:
        return value is seal.callback
    return (
        isinstance(value, MethodType)
        and value.__self__ is seal.bound_self
        and value.__func__ is seal.bound_function
    )


def _require_sha256(value: object) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(_EXECUTION_ERROR)
    return value


def _plain_json(value: JSONValue) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _snapshot_action(value: object) -> SemanticAction:
    if not _semantic_action_has_canonical_shape(value):
        raise ValueError(_EXECUTION_ERROR)
    action = cast(SemanticAction, value)
    rebuilt = SemanticAction(
        object.__getattribute__(action, "name"),
        object.__getattribute__(action, "arguments"),
    )
    if action != rebuilt:
        raise ValueError(_EXECUTION_ERROR)
    return rebuilt


def _snapshot_native_call(
    value: object,
) -> tuple[Mapping[str, JSONValue], str, str, dict[str, object]]:
    try:
        call = _freeze_mapping(value, "AppWorld native call")
        if set(call) != _CALL_KEYS:
            raise ValueError
        name = call.get("name")
        arguments = call.get("arguments")
        if type(name) is not str or not isinstance(arguments, Mapping):
            raise ValueError
        app_name, separator, api_name = name.partition("__")
        if (
            separator != "__"
            or not app_name
            or not api_name
            or not app_name.isascii()
            or not api_name.isascii()
            or not app_name.isidentifier()
            or not api_name.isidentifier()
            or set(arguments) & _REQUESTER_CONTROL_NAMES
        ):
            raise ValueError
        plain_arguments = _plain_json(cast(JSONValue, arguments))
        if type(plain_arguments) is not dict:
            raise ValueError
        return call, app_name, api_name, plain_arguments
    except Exception:
        raise ValueError(_EXECUTION_ERROR) from None


def _snapshot_requester_observation(value: object) -> JSONValue:
    if type(value) not in (dict, list):
        raise ValueError(_EXECUTION_ERROR)
    try:
        return _freeze_json_root(value, "AppWorld requester observation")
    except Exception:
        raise ValueError(_EXECUTION_ERROR) from None


def _require_json_unchanged(value: object, expected: JSONValue) -> None:
    try:
        if canonical_json_bytes(cast(JSONValue, value)) != canonical_json_bytes(expected):
            raise _ExecutorIntegrityError
    except _ExecutorIntegrityError:
        raise
    except BaseException:
        raise _ExecutorIntegrityError from None


def _require_action_unchanged(value: object, expected: SemanticAction) -> None:
    try:
        if _snapshot_action(value) != expected:
            raise _ExecutorIntegrityError
    except _ExecutorIntegrityError:
        raise
    except BaseException:
        raise _ExecutorIntegrityError from None


def _snapshot_trace(value: object) -> ExecutionTrace:
    try:
        if not _execution_trace_has_canonical_shape(value):
            raise _ExecutorIntegrityError
        trace = cast(ExecutionTrace, value)
        rebuilt = ExecutionTrace(
            object.__getattribute__(trace, "surface_calls"),
            tuple(
                _snapshot_action(action)
                for action in object.__getattribute__(trace, "semantic_actions")
            ),
            object.__getattribute__(trace, "base_calls"),
        )
        if trace != rebuilt:
            raise _ExecutorIntegrityError
        return rebuilt
    except _ExecutorIntegrityError:
        raise
    except BaseException:
        raise _ExecutorIntegrityError from None


def _compact_evaluator_score(tracker: object) -> Mapping[str, JSONValue]:
    try:
        pass_count = tracker.pass_count
        fail_count = tracker.fail_count
        total_count = tracker.total_count
        num_tests = tracker.num_tests
        success = tracker.success
        counts = (pass_count, fail_count, total_count, num_tests)
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError
        if total_count != pass_count + fail_count or total_count > num_tests:
            raise ValueError
        if type(success) is not bool or success != (pass_count == num_tests):
            raise ValueError
        return _freeze_mapping(
            {
                "pass_count": pass_count,
                "fail_count": fail_count,
                "total_count": total_count,
                "num_tests": num_tests,
                "success": success,
            },
            "AppWorld evaluator score",
        )
    except Exception:
        raise ValueError(_EXECUTION_ERROR) from None


def _effect_digest(before_state_sha256: str, after_state_sha256: str) -> str:
    return hashlib.sha256(
        _EFFECT_DOMAIN
        + canonical_json_bytes(
            {
                "before_state_sha256": before_state_sha256,
                "after_state_sha256": after_state_sha256,
            }
        )
    ).hexdigest()


def _transition_digest(
    effects: tuple[PhysicalCallEffect, ...],
    final_state_sha256: str,
    evaluator_digest: str,
) -> str:
    return hashlib.sha256(
        _TRANSITION_DOMAIN
        + canonical_json_bytes(
            {
                "effects": [
                    {
                        "call_index": effect.call_index,
                        "base_call_fingerprint": effect.base_call_fingerprint,
                        "effect_digest": effect.effect_digest,
                    }
                    for effect in effects
                ],
                "final_state_sha256": final_state_sha256,
                "evaluator_digest": evaluator_digest,
            }
        )
    ).hexdigest()


class AppWorldEpisodeExecutor:
    """Execute one non-empty direct-replay plan against one injected live world."""

    __slots__ = (
        "_adapter",
        "_adapter_method_seals",
        "_adapter_method_seals_seal",
        "_adapter_seal",
        "_adapter_variant_seal",
        "_binding_seals",
        "_binding_seals_seal",
        "_callable_seal_entries",
        "_callable_seal_entries_seal",
        "_entry_latch",
        "_entry_latch_seal",
        "_evaluate_seal",
        "_evaluator_hasher",
        "_evaluator_hasher_seal",
        "_integrity_token",
        "_models_seal",
        "_phase",
        "_request_seal",
        "_requester_seal",
        "_save_seal",
        "_started",
        "_state_hasher",
        "_state_hasher_seal",
        "_world",
        "_world_seal",
    )

    def __init__(
        self,
        world: object,
        adapter: SemanticAdapter,
        *,
        state_hasher: Callable[[object], str] = canonical_state_sha256,
        evaluator_hasher: Callable[[object], str] = evaluator_sha256,
    ) -> None:
        try:
            if not isinstance(adapter, SemanticAdapter):
                raise ValueError
            requester = world.requester
            models = world.models
            request_seal = _make_callable_seal(requester.request)
            save_seal = _make_callable_seal(world.save)
            evaluate_seal = _make_callable_seal(world.evaluate)
            state_hasher_seal = _make_callable_seal(state_hasher)
            evaluator_hasher_seal = _make_callable_seal(evaluator_hasher)
            adapter_methods = tuple(
                (name, _make_callable_seal(getattr(adapter, name)))
                for name in (
                    "surface_to_semantic",
                    "semantic_to_base_calls",
                    "base_observation_to_surface",
                    "canonicalize_trace",
                )
            )
            variant = adapter.variant
        except Exception:
            raise ValueError(_CONFIGURATION_ERROR) from None

        self._world = world
        self._world_seal = world
        self._requester_seal = requester
        self._models_seal = models
        self._adapter = adapter
        self._adapter_seal = adapter
        self._adapter_variant_seal = variant
        self._request_seal = request_seal
        self._save_seal = save_seal
        self._evaluate_seal = evaluate_seal
        self._state_hasher = state_hasher
        self._state_hasher_seal = state_hasher_seal
        self._evaluator_hasher = evaluator_hasher
        self._evaluator_hasher_seal = evaluator_hasher_seal
        self._adapter_method_seals = adapter_methods
        self._adapter_method_seals_seal = adapter_methods
        binding_seals = (
            request_seal,
            save_seal,
            evaluate_seal,
            state_hasher_seal,
            evaluator_hasher_seal,
        )
        self._binding_seals = binding_seals
        self._binding_seals_seal = binding_seals
        callable_seal_entries = tuple(
            (seal, seal.callback, seal.bound_self, seal.bound_function)
            for seal in (
                *binding_seals,
                *(seal for _, seal in adapter_methods),
            )
        )
        self._callable_seal_entries = callable_seal_entries
        self._callable_seal_entries_seal = callable_seal_entries
        entry_latch = iter((_ENTRY_LATCH_TOKEN,))
        self._entry_latch = entry_latch
        self._entry_latch_seal = entry_latch
        self._started = False
        self._phase = _READY
        self._integrity_token = _INTEGRITY_TOKEN
        try:
            self._require_integrity()
        except _ExecutorIntegrityError:
            raise ValueError(_CONFIGURATION_ERROR) from None

    def _require_integrity(self) -> None:
        try:
            world = object.__getattribute__(self, "_world")
            adapter = object.__getattribute__(self, "_adapter")
            requester = object.__getattribute__(self, "_requester_seal")
            binding_seals = object.__getattribute__(self, "_binding_seals")
            adapter_method_seals = object.__getattribute__(
                self,
                "_adapter_method_seals",
            )
            callable_seal_entries = object.__getattribute__(
                self,
                "_callable_seal_entries",
            )
            started = object.__getattribute__(self, "_started")
            phase = object.__getattribute__(self, "_phase")
            entry_latch = object.__getattribute__(self, "_entry_latch")
            entry_latch_seal = object.__getattribute__(self, "_entry_latch_seal")
            if entry_latch is not entry_latch_seal or type(entry_latch) is not _TUPLE_ITERATOR_TYPE:
                raise _ExecutorIntegrityError
            entry_latch_length = entry_latch.__length_hint__()
            if (
                object.__getattribute__(self, "_integrity_token") is not _INTEGRITY_TOKEN
                or world is not object.__getattribute__(self, "_world_seal")
                or adapter is not object.__getattribute__(self, "_adapter_seal")
                or binding_seals is not object.__getattribute__(self, "_binding_seals_seal")
                or type(binding_seals) is not tuple
                or len(binding_seals) != 5
                or object.__getattribute__(self, "_request_seal") is not binding_seals[0]
                or object.__getattribute__(self, "_save_seal") is not binding_seals[1]
                or object.__getattribute__(self, "_evaluate_seal") is not binding_seals[2]
                or object.__getattribute__(self, "_state_hasher_seal") is not binding_seals[3]
                or object.__getattribute__(self, "_evaluator_hasher_seal") is not binding_seals[4]
                or adapter_method_seals
                is not object.__getattribute__(self, "_adapter_method_seals_seal")
                or type(adapter_method_seals) is not tuple
                or len(adapter_method_seals) != 4
                or callable_seal_entries
                is not object.__getattribute__(self, "_callable_seal_entries_seal")
                or type(callable_seal_entries) is not tuple
                or len(callable_seal_entries) != 9
                or world.requester is not requester
                or world.models is not object.__getattribute__(self, "_models_seal")
                or adapter.variant is not object.__getattribute__(self, "_adapter_variant_seal")
                or not _callable_matches(
                    requester.request,
                    object.__getattribute__(self, "_request_seal"),
                )
                or not _callable_matches(
                    world.save,
                    object.__getattribute__(self, "_save_seal"),
                )
                or not _callable_matches(
                    world.evaluate,
                    object.__getattribute__(self, "_evaluate_seal"),
                )
                or object.__getattribute__(self, "_state_hasher")
                is not object.__getattribute__(self, "_state_hasher_seal").callback
                or object.__getattribute__(self, "_evaluator_hasher")
                is not object.__getattribute__(self, "_evaluator_hasher_seal").callback
                or type(entry_latch_length) is not int
                or type(started) is not bool
                or (phase, started, entry_latch_length)
                not in {
                    (_READY, False, 1),
                    (_RUNNING, True, 0),
                    (_CONSUMED, True, 0),
                }
            ):
                raise _ExecutorIntegrityError
            current_callable_seals = (
                *binding_seals,
                *(seal for _, seal in adapter_method_seals),
            )
            for seal, entry in zip(
                current_callable_seals,
                callable_seal_entries,
                strict=True,
            ):
                if (
                    type(seal) is not _CallableSeal
                    or type(entry) is not tuple
                    or len(entry) != 4
                    or seal is not entry[0]
                    or seal.callback is not entry[1]
                    or seal.bound_self is not entry[2]
                    or seal.bound_function is not entry[3]
                    or not callable(seal.callback)
                    or (
                        seal.bound_self is None
                        and (
                            seal.bound_function is not None or isinstance(seal.callback, MethodType)
                        )
                    )
                    or (
                        seal.bound_self is not None
                        and not (
                            isinstance(seal.callback, MethodType)
                            and seal.callback.__self__ is seal.bound_self
                            and seal.callback.__func__ is seal.bound_function
                        )
                    )
                ):
                    raise _ExecutorIntegrityError
            for name, seal in adapter_method_seals:
                if not _callable_matches(getattr(adapter, name), seal):
                    raise _ExecutorIntegrityError
        except _ExecutorIntegrityError:
            raise
        except BaseException:
            raise _ExecutorIntegrityError from None

    def _invoke(
        self,
        callback: Callable[..., _ResultT],
        *args: object,
        **kwargs: object,
    ) -> _ResultT:
        self._require_integrity()
        if object.__getattribute__(self, "_phase") != _RUNNING:
            raise _ExecutorIntegrityError
        try:
            return callback(*args, **kwargs)
        finally:
            self._require_integrity()
            if object.__getattribute__(self, "_phase") != _RUNNING:
                raise _ExecutorIntegrityError

    def _hash_state(self) -> str:
        models = object.__getattribute__(self, "_models_seal")
        callback = cast(
            Callable[[object], object],
            object.__getattribute__(self, "_state_hasher_seal").callback,
        )
        return _require_sha256(self._invoke(callback, models))

    def _execute(self, surface_calls: Sequence[Mapping[str, JSONValue]]) -> _EpisodeRecord:
        if isinstance(surface_calls, (str, bytes, bytearray)) or not isinstance(
            surface_calls, Sequence
        ):
            raise ValueError(_EXECUTION_ERROR)
        raw_calls = tuple(surface_calls)
        if not raw_calls:
            raise ValueError(_EXECUTION_ERROR)
        surface_call_snapshots = tuple(
            _freeze_mapping(call, "AppWorld surface call") for call in raw_calls
        )
        self._require_integrity()

        adapter_method_seals = object.__getattribute__(
            self,
            "_adapter_method_seals_seal",
        )
        parse = adapter_method_seals[0][1].callback
        compile_action_callback = adapter_method_seals[1][1].callback
        wrap_observation = adapter_method_seals[2][1].callback
        canonicalize = adapter_method_seals[3][1].callback
        request = object.__getattribute__(self, "_request_seal").callback
        save = object.__getattribute__(self, "_save_seal").callback
        steps: list[_ExecutedStep] = []
        previous_post_state: str | None = None

        for surface_call in surface_call_snapshots:
            parse_surface_call = _freeze_mapping(
                surface_call,
                "AppWorld parse surface call",
            )
            raw_actions = self._invoke(
                parse,
                parse_surface_call,
            )
            _require_json_unchanged(parse_surface_call, cast(JSONValue, surface_call))
            if type(raw_actions) is not tuple or len(raw_actions) != 1:
                raise ValueError(_EXECUTION_ERROR)
            action = _snapshot_action(raw_actions[0])

            compile_action = _snapshot_action(action)
            raw_base_calls = self._invoke(
                compile_action_callback,
                compile_action,
            )
            _require_action_unchanged(compile_action, action)
            if type(raw_base_calls) is not tuple or len(raw_base_calls) != 1:
                raise ValueError(_EXECUTION_ERROR)
            base_call, app_name, api_name, plain_arguments = _snapshot_native_call(
                raw_base_calls[0]
            )
            if base_call["name"] != action.name:
                raise ValueError(_EXECUTION_ERROR)

            before_state = self._hash_state()
            if previous_post_state is not None and before_state != previous_post_state:
                raise ValueError(_EXECUTION_ERROR)
            raw_observation = self._invoke(
                request,
                _app_name=app_name,
                _api_name=api_name,
                **plain_arguments,
            )
            self._invoke(save)
            after_state = self._hash_state()
            base_observation = _snapshot_requester_observation(raw_observation)
            self._require_integrity()
            wrap_surface_call = _freeze_mapping(
                surface_call,
                "AppWorld wrap surface call",
            )
            wrap_action = _snapshot_action(action)
            wrap_base_observation = _freeze_json_root(
                base_observation,
                "AppWorld wrap base observation",
            )
            raw_surface_observation = self._invoke(
                wrap_observation,
                wrap_surface_call,
                (wrap_action,),
                ((wrap_base_observation,),),
            )
            _require_json_unchanged(wrap_surface_call, cast(JSONValue, surface_call))
            _require_action_unchanged(wrap_action, action)
            _require_json_unchanged(wrap_base_observation, base_observation)
            surface_observation = _freeze_json_root(
                raw_surface_observation,
                "AppWorld surface observation",
            )
            self._require_integrity()
            steps.append(
                _ExecutedStep(
                    surface_call,
                    action,
                    base_call,
                    base_observation,
                    surface_observation,
                    before_state,
                    after_state,
                )
            )
            previous_post_state = after_state

        trace = ExecutionTrace(
            tuple(step.surface_call for step in steps),
            tuple(step.action for step in steps),
            tuple(step.base_call for step in steps),
        )
        canonicalize_trace = _snapshot_trace(trace)
        canonical_actions = self._invoke(
            canonicalize,
            canonicalize_trace,
        )
        if _snapshot_trace(canonicalize_trace) != trace:
            raise _ExecutorIntegrityError
        if type(canonical_actions) is not tuple or len(canonical_actions) != len(steps):
            raise ValueError(_EXECUTION_ERROR)
        canonical_snapshots = tuple(_snapshot_action(action) for action in canonical_actions)
        if canonical_snapshots != tuple(step.action for step in steps):
            raise ValueError(_EXECUTION_ERROR)

        evaluate = object.__getattribute__(self, "_evaluate_seal").callback
        tracker = self._invoke(evaluate, suppress_errors=False)
        evaluator_score = _compact_evaluator_score(tracker)
        self._require_integrity()
        evaluator_hasher = cast(
            Callable[[object], object],
            object.__getattribute__(self, "_evaluator_hasher_seal").callback,
        )
        evaluator_digest = _require_sha256(self._invoke(evaluator_hasher, tracker))
        if evaluator_score != _compact_evaluator_score(tracker):
            raise ValueError(_EXECUTION_ERROR)
        self._require_integrity()

        effects = tuple(
            PhysicalCallEffect(
                index,
                base_call_fingerprint(step.base_call),
                _effect_digest(step.before_state_sha256, step.after_state_sha256),
            )
            for index, step in enumerate(steps)
        )
        final_state = cast(str, previous_post_state)
        return _EpisodeRecord(
            tuple(steps),
            trace,
            steps[0].before_state_sha256,
            final_state,
            evaluator_score,
            evaluator_digest,
            effects,
            _transition_digest(effects, final_state, evaluator_digest),
        )

    def execute_plan(
        self,
        surface_calls: Sequence[Mapping[str, JSONValue]],
    ) -> _EpisodeRecord:
        """Execute one non-empty plan and permanently consume this executor."""

        try:
            self._require_integrity()
        except _ExecutorIntegrityError:
            consumed_latch = iter(())
            object.__setattr__(self, "_entry_latch", consumed_latch)
            object.__setattr__(self, "_entry_latch_seal", consumed_latch)
            object.__setattr__(self, "_started", True)
            object.__setattr__(self, "_phase", _CONSUMED)
            raise ValueError(_INTEGRITY_ERROR) from None
        if object.__getattribute__(self, "_phase") != _READY:
            raise ValueError(_EXECUTION_ERROR)
        entry_latch = object.__getattribute__(self, "_entry_latch")
        if next(entry_latch, None) is not _ENTRY_LATCH_TOKEN:
            object.__setattr__(self, "_started", True)
            object.__setattr__(self, "_phase", _CONSUMED)
            raise ValueError(_INTEGRITY_ERROR) from None
        object.__setattr__(self, "_started", True)
        object.__setattr__(self, "_phase", _RUNNING)
        try:
            try:
                return self._execute(surface_calls)
            except _ExecutorIntegrityError:
                raise ValueError(_INTEGRITY_ERROR) from None
            except Exception:
                raise ValueError(_EXECUTION_ERROR) from None
        finally:
            try:
                phase_was_running = object.__getattribute__(self, "_phase") == _RUNNING
                started_was_true = object.__getattribute__(self, "_started") is True
            except BaseException:
                phase_was_running = False
                started_was_true = False
            object.__setattr__(self, "_started", True)
            object.__setattr__(self, "_phase", _CONSUMED)
            try:
                self._require_integrity()
            except _ExecutorIntegrityError:
                raise ValueError(_INTEGRITY_ERROR) from None
            if not phase_was_running or not started_was_true:
                raise ValueError(_INTEGRITY_ERROR) from None


__all__ = ["AppWorldEpisodeExecutor"]
