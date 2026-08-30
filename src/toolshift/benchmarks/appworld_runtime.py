"""Isolated, in-process execution of validated AppWorld native calls.

The module deliberately has no import-time dependency on AppWorld.  A caller
owns the lifecycle of the injected live world; this executor is one-shot and
never closes, resets, or rolls that world back.
"""

from __future__ import annotations

import hashlib
import inspect
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
_ORACLE_CAPTURE_ERROR = "AppWorld oracle capture failed"
_ORACLE_INTEGRITY_ERROR = "AppWorld oracle capture integrity validation failed"
_EXECUTION_FAILURE_PREFIX = "Execution failed. Traceback:"
_CALL_KEYS = frozenset({"name", "arguments"})
_CAPTURE_METHOD_NAMES = ("request", "get", "post", "put", "patch", "delete")
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


class _OracleIntegrityError(ValueError):
    """Internal oracle-capture marker converted to a static public error."""


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


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _CapturedOraclePlan:
    native_calls: tuple[Mapping[str, JSONValue], ...]

    def __post_init__(self) -> None:
        try:
            if type(self.native_calls) is not tuple or not self.native_calls:
                raise ValueError
            rebuilt = tuple(_snapshot_native_call(call)[0] for call in self.native_calls)
        except BaseException:
            raise ValueError(_ORACLE_CAPTURE_ERROR) from None
        object.__setattr__(self, "native_calls", rebuilt)

    def __reduce__(self) -> object:
        raise TypeError("AppWorld oracle plans cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("AppWorld oracle plans cannot be serialized")


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


def _capture_oracle_plan(world: object) -> _CapturedOraclePlan:
    """Capture calls from the pinned trusted oracle's ordinary instance dispatch.

    Exact-instance patches cannot generally intercept an explicit
    ``type(requester).verb(...)`` call or a previously cached bound alias.  The
    private compatibility probe must verify that the pinned oracle makes no such
    untracked bypass; this function deliberately performs no class/global patch.
    """

    primary_error: BaseException | None = None
    result: _CapturedOraclePlan | None = None
    final_integrity_failure = False
    post_capture_finalizer: Callable[[], bool] | None = None
    try:
        requester = world.requester
        raw_instance_slots = object.__getattribute__(requester, "__dict__")
        if type(raw_instance_slots) is not dict:
            raise ValueError
        raw_states = {
            name: (name in raw_instance_slots, raw_instance_slots.get(name))
            for name in _CAPTURE_METHOD_NAMES
        }
        original_seals = {
            name: _make_callable_seal(getattr(requester, name)) for name in _CAPTURE_METHOD_NAMES
        }
        execute_seal = _make_callable_seal(world.execute)
        evaluate_seal = _make_callable_seal(world.evaluate)
        task = world.task
        ground_truth = task.ground_truth
        compiled_solution_code = ground_truth.compiled_solution_code
        if type(compiled_solution_code) is not str:
            raise ValueError
        tracker_root = requester.requests
        if type(tracker_root) is not list:
            raise ValueError
        initial_tracker_length = len(tracker_root)
        all_original_seals = (*original_seals.values(), execute_seal, evaluate_seal)
        seal_entries = tuple(
            (seal, seal.callback, seal.bound_self, seal.bound_function)
            for seal in all_original_seals
        )

        request_parameters = tuple(
            inspect.signature(original_seals["request"].callback).parameters.values()
        )
        if (
            tuple(parameter.name for parameter in request_parameters)
            != (
                "_app_name",
                "_api_name",
                "client",
                "raise_on_failure",
                "show",
                "track",
                "data",
            )
            or tuple(parameter.kind for parameter in request_parameters)
            != (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_KEYWORD,
            )
            or request_parameters[0].default is not inspect.Parameter.empty
            or request_parameters[1].default is not inspect.Parameter.empty
            or request_parameters[2].default is not None
            or request_parameters[3].default is not None
            or request_parameters[4].default is not False
            or request_parameters[5].default is not True
        ):
            raise ValueError

        captured_calls: list[Mapping[str, JSONValue]] = []
        active_capability: list[object | None] = [None]
        sticky_violation = [False]
        low_level_capability = object()
        patch_values: dict[str, object] = {}
        patch_seals: dict[str, _CallableSeal] = {}

        def require_seal_shape(
            seal: _CallableSeal,
            entry: tuple[object, object, object, object],
        ) -> None:
            if (
                type(seal) is not _CallableSeal
                or seal is not entry[0]
                or seal.callback is not entry[1]
                or seal.bound_self is not entry[2]
                or seal.bound_function is not entry[3]
            ):
                raise _OracleIntegrityError

        def require_integrity(*, patched: bool) -> None:
            try:
                if (
                    world.requester is not requester
                    or world.task is not task
                    or task.ground_truth is not ground_truth
                    or ground_truth.compiled_solution_code is not compiled_solution_code
                    or object.__getattribute__(requester, "__dict__") is not raw_instance_slots
                    or requester.requests is not tracker_root
                    or type(tracker_root) is not list
                    or not _callable_matches(world.execute, execute_seal)
                    or not _callable_matches(world.evaluate, evaluate_seal)
                ):
                    raise _OracleIntegrityError
                for seal, entry in zip(all_original_seals, seal_entries, strict=True):
                    require_seal_shape(seal, entry)
                expected_seals = patch_seals if patched else original_seals
                expected_values = patch_values if patched else raw_states
                for name in _CAPTURE_METHOD_NAMES:
                    if patched:
                        if raw_instance_slots.get(name) is not expected_values[
                            name
                        ] or not _callable_matches(
                            getattr(requester, name),
                            expected_seals[name],
                        ):
                            raise _OracleIntegrityError
                    else:
                        existed, raw_value = expected_values[name]
                        if (
                            (name in raw_instance_slots) != existed
                            or (existed and raw_instance_slots.get(name) is not raw_value)
                            or not _callable_matches(
                                getattr(requester, name),
                                expected_seals[name],
                            )
                        ):
                            raise _OracleIntegrityError
            except _OracleIntegrityError:
                raise
            except BaseException:
                raise _OracleIntegrityError from None

        def finalize_restored_methods() -> bool:
            integrity_failed = False
            try:
                require_integrity(patched=False)
            except BaseException:
                integrity_failed = True
            for name in reversed(_CAPTURE_METHOD_NAMES):
                try:
                    existed, raw_value = raw_states[name]
                    current_exists = name in raw_instance_slots
                    current_value = raw_instance_slots.get(name)
                    if existed and (not current_exists or current_value is not raw_value):
                        setattr(requester, name, raw_value)
                    elif not existed and current_exists:
                        delattr(requester, name)
                except BaseException:
                    integrity_failed = True
            try:
                require_integrity(patched=False)
            except BaseException:
                integrity_failed = True
            return integrity_failed

        post_capture_finalizer = finalize_restored_methods

        def guarded_verb(name: str) -> Callable[..., object]:
            original = original_seals[name].callback

            def guard(self: object, *args: object, **kwargs: object) -> object:
                if self is not requester or active_capability[0] is not low_level_capability:
                    sticky_violation[0] = True
                    raise _OracleIntegrityError
                require_integrity(patched=True)
                try:
                    return original(*args, **kwargs)
                finally:
                    require_integrity(patched=True)

            return guard

        def capture_request(
            self: object,
            _app_name: object,
            _api_name: object,
            client: object = None,
            raise_on_failure: object = None,
            show: object = False,
            track: object = True,
            **data: object,
        ) -> object:
            if (
                self is not requester
                or active_capability[0] is not None
                or track is not True
                or "_system_datetime" in data
                or type(_app_name) is not str
                or type(_api_name) is not str
            ):
                sticky_violation[0] = True
                raise _OracleIntegrityError
            try:
                native_call, _, _, _ = _snapshot_native_call(
                    {"name": _app_name + "__" + _api_name, "arguments": data}
                )
            except BaseException:
                sticky_violation[0] = True
                raise _OracleIntegrityError from None
            require_integrity(patched=True)
            before_length = len(tracker_root)
            active_capability[0] = low_level_capability
            try:
                call_result = original_seals["request"].callback(
                    _app_name,
                    _api_name,
                    client=client,
                    raise_on_failure=raise_on_failure,
                    show=show,
                    track=track,
                    **data,
                )
            finally:
                active_capability[0] = None
            require_integrity(patched=True)
            if len(tracker_root) != before_length + 1:
                raise _OracleIntegrityError
            captured_calls.append(native_call)
            return call_result

        patch_values["request"] = MethodType(capture_request, requester)
        for verb_name in _CAPTURE_METHOD_NAMES[1:]:
            patch_values[verb_name] = MethodType(guarded_verb(verb_name), requester)
        patch_seals = {name: _make_callable_seal(value) for name, value in patch_values.items()}

        execution_output: object | None = None
        execution_error: BaseException | None = None
        attempted_patches: list[str] = []
        patching_complete = False
        restoration_failed = False
        try:
            require_integrity(patched=False)
            for name in _CAPTURE_METHOD_NAMES:
                attempted_patches.append(name)
                setattr(requester, name, patch_values[name])
                if raw_instance_slots.get(name) is not patch_values[name]:
                    raise _OracleIntegrityError
            patching_complete = True
            require_integrity(patched=True)
            execution_output = execute_seal.callback(
                compiled_solution_code + "\nsolution(apis, requester)"
            )
        except BaseException as error:
            execution_error = error
        finally:
            if patching_complete:
                try:
                    require_integrity(patched=True)
                except BaseException:
                    restoration_failed = True
            for name in reversed(attempted_patches):
                try:
                    existed, raw_value = raw_states[name]
                    if existed:
                        setattr(requester, name, raw_value)
                    elif name in raw_instance_slots:
                        delattr(requester, name)
                except BaseException:
                    restoration_failed = True
            try:
                require_integrity(patched=False)
            except BaseException:
                restoration_failed = True

        if restoration_failed:
            raise _OracleIntegrityError
        if execution_error is not None:
            raise execution_error
        if sticky_violation[0]:
            raise _OracleIntegrityError
        if (
            type(execution_output) is not str
            or not execution_output
            or execution_output.startswith(_EXECUTION_FAILURE_PREFIX)
            or not captured_calls
            or len(tracker_root) != initial_tracker_length + len(captured_calls)
        ):
            raise ValueError
        require_integrity(patched=False)
        try:
            tracker = evaluate_seal.callback(suppress_errors=False)
        finally:
            require_integrity(patched=False)
        try:
            evaluator_score = _compact_evaluator_score(tracker)
        finally:
            require_integrity(patched=False)
        if evaluator_score["success"] is not True:
            raise ValueError
        result = _CapturedOraclePlan(tuple(captured_calls))
    except BaseException as error:
        primary_error = error
    finally:
        if post_capture_finalizer is not None:
            try:
                final_integrity_failure = post_capture_finalizer()
            except BaseException:
                final_integrity_failure = True

    if final_integrity_failure or isinstance(primary_error, _OracleIntegrityError):
        raise ValueError(_ORACLE_INTEGRITY_ERROR) from None
    if primary_error is not None:
        if isinstance(primary_error, Exception):
            raise ValueError(_ORACLE_CAPTURE_ERROR) from None
        raise primary_error
    if result is None:
        raise ValueError(_ORACLE_CAPTURE_ERROR) from None
    return result


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
