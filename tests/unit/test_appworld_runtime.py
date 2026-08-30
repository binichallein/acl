"""Synthetic tests for the isolated AppWorld direct-replay executor."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.trace import base_call_fingerprint
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    canonical_json_bytes,
)

_EFFECT_DOMAIN = b"toolshift.appworld.effect.v1\0"
_TRANSITION_DOMAIN = b"toolshift.appworld.transition.v1\0"
_EXECUTION_ERROR = "AppWorld episode execution failed"
_INTEGRITY_ERROR = "AppWorld executor integrity validation failed"


def _sha(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


class _Tracker:
    def __init__(
        self,
        *,
        pass_count: int = 2,
        fail_count: int = 0,
        num_tests: int = 2,
        success: bool = True,
    ) -> None:
        self.pass_count = pass_count
        self.fail_count = fail_count
        self.total_count = pass_count + fail_count
        self.num_tests = num_tests
        self.success = success


class _Models:
    def __init__(self) -> None:
        self.state = 0


class _Requester:
    def __init__(
        self,
        world: _World,
        events: list[str],
        responses: list[object],
    ) -> None:
        self.world = world
        self.events = events
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.inside_request = False
        self.hook: Callable[[], None] | None = None

    def request(self, **kwargs: object) -> object:
        self.inside_request = True
        try:
            self.events.append("request")
            self.calls.append(kwargs)
            if self.hook is not None:
                self.hook()
            self.world.models.state += 1
            return self.responses.pop(0)
        finally:
            self.inside_request = False


class _World:
    def __init__(
        self,
        events: list[str],
        responses: list[object] | None = None,
        *,
        tracker: _Tracker | None = None,
    ) -> None:
        self.events = events
        self.models = _Models()
        self.requester = _Requester(
            self,
            events,
            responses or [{"ok": True}, {"ok": True}],
        )
        self.tracker = tracker or _Tracker()
        self.save_calls = 0
        self.evaluate_calls: list[bool] = []
        self.save_hook: Callable[[], None] | None = None
        self.evaluate_hook: Callable[[], None] | None = None

    def save(self) -> None:
        assert self.requester.inside_request is False
        self.events.append("save")
        self.save_calls += 1
        if self.save_hook is not None:
            self.save_hook()

    def evaluate(self, *, suppress_errors: bool) -> _Tracker:
        self.events.append("evaluate")
        self.evaluate_calls.append(suppress_errors)
        if self.evaluate_hook is not None:
            self.evaluate_hook()
        return self.tracker

    def execute(self, code: str) -> str:  # pragma: no cover - must never be called.
        raise AssertionError(code)


class _ReturningSaveWorld(_World):
    def save(self) -> object:
        super().save()
        return object()


class _Adapter(SemanticAdapter):
    def __init__(self, events: list[str]) -> None:
        super().__init__(
            SchemaVariant(
                "synthetic-appworld-runtime",
                (
                    SurfaceToolSpec(
                        "notes__create_note",
                        "Synthetic.",
                        {
                            "type": "object",
                            "properties": {"value": {"type": "integer"}},
                            "required": ["value"],
                            "additionalProperties": False,
                        },
                    ),
                ),
                {"kind": "synthetic"},
            )
        )
        self.events = events
        self.parse_hook: Callable[[], None] | None = None
        self.compile_hook: Callable[[], None] | None = None
        self.wrap_hook: Callable[[], None] | None = None
        self.canonicalize_hook: Callable[[], None] | None = None
        self.action_count = 1
        self.base_call_count = 1

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        self.events.append("parse")
        if self.parse_hook is not None:
            self.parse_hook()
        if self.action_count == 0:
            return ()
        actions = (SemanticAction("notes__create_note", surface_call["arguments"]),)
        return actions * self.action_count

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        self.events.append("compile")
        if self.compile_hook is not None:
            self.compile_hook()
        call: Mapping[str, JSONValue] = {
            "name": action.name,
            "arguments": action.arguments,
        }
        return (call,) * self.base_call_count

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self.events.append("wrap")
        if self.wrap_hook is not None:
            self.wrap_hook()
        assert len(actions) == 1
        assert len(base_observation_groups) == 1
        assert len(base_observation_groups[0]) == 1
        return {"wrapped": base_observation_groups[0][0]}

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self.events.append("canonicalize")
        if self.canonicalize_hook is not None:
            self.canonicalize_hook()
        return trace.semantic_actions


class _InputMutatingAdapter(_Adapter):
    def __init__(self, events: list[str], phase: str) -> None:
        super().__init__(events)
        self.phase = phase

    def surface_to_semantic(
        self,
        surface_call: Mapping[str, JSONValue],
    ) -> tuple[SemanticAction, ...]:
        actions = super().surface_to_semantic(surface_call)
        if self.phase == "parse":
            object.__setattr__(surface_call, "_items", ())
        return actions

    def semantic_to_base_calls(
        self,
        action: SemanticAction,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        calls = super().semantic_to_base_calls(action)
        if self.phase == "compile":
            object.__setattr__(action, "_arguments_canonical", b"PRIVATE_ACTION_CACHE")
        return calls

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        observation = super().base_observation_to_surface(
            surface_call,
            actions,
            base_observation_groups,
        )
        if self.phase == "wrap":
            object.__setattr__(base_observation_groups[0][0], "_items", ())
        return observation

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        actions = super().canonicalize_trace(trace)
        if self.phase == "canonicalize":
            object.__setattr__(trace, "_surface_calls_canonical", b"PRIVATE_TRACE_CACHE")
        return actions


class _OneReadForgingAdapter(_Adapter):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.armed = False

    def __getattribute__(self, name: str) -> object:
        if (
            name == "surface_to_semantic"
            and object.__getattribute__(self, "armed")
            and sys._getframe(1).f_code.co_name == "_execute"
        ):

            def forged(call: Mapping[str, JSONValue]) -> tuple[SemanticAction, ...]:
                object.__getattribute__(self, "events").append("forged-parse")
                return (SemanticAction("notes__create_note", call["arguments"]),)

            return forged
        return super().__getattribute__(name)


def _surface_call(value: int) -> dict[str, JSONValue]:
    return {"name": "notes__create_note", "arguments": {"value": value}}


def _state_hasher(events: list[str]) -> Callable[[object], str]:
    def hash_state(models: object) -> str:
        assert type(models) is _Models
        events.append(f"hash:{models.state}")
        return _sha(models.state)

    return hash_state


def _evaluator_hasher(events: list[str]) -> Callable[[object], str]:
    def hash_evaluator(tracker: object) -> str:
        assert type(tracker) is _Tracker
        events.append("evaluator-hash")
        return _sha(
            (
                tracker.pass_count,
                tracker.fail_count,
                tracker.total_count,
                tracker.num_tests,
                tracker.success,
            )
        )

    return hash_evaluator


def _executor(
    world: _World,
    adapter: _Adapter,
    events: list[str],
):
    from toolshift.benchmarks.appworld_runtime import AppWorldEpisodeExecutor

    return AppWorldEpisodeExecutor(
        world,
        adapter,
        state_hasher=_state_hasher(events),
        evaluator_hasher=_evaluator_hasher(events),
    )


def test_runtime_module_has_no_import_time_appworld_dependency() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (
        "import sys; "
        "assert 'appworld' not in sys.modules; "
        "import toolshift.benchmarks.appworld_runtime; "
        "assert 'appworld' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_execute_plan_runs_two_calls_in_the_fixed_physical_order() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": [1]}, [{"ok": 2}]])
    adapter = _Adapter(events)

    record = _executor(world, adapter, events).execute_plan([_surface_call(1), _surface_call(2)])

    assert events == [
        "parse",
        "compile",
        "hash:0",
        "request",
        "save",
        "hash:1",
        "wrap",
        "parse",
        "compile",
        "hash:1",
        "request",
        "save",
        "hash:2",
        "wrap",
        "canonicalize",
        "evaluate",
        "evaluator-hash",
    ]
    assert world.requester.calls == [
        {"_app_name": "notes", "_api_name": "create_note", "value": 1},
        {"_app_name": "notes", "_api_name": "create_note", "value": 2},
    ]
    assert world.save_calls == 2
    assert world.evaluate_calls == [False]
    assert len(record.steps) == 2
    assert record.initial_state_sha256 == _sha(0)
    assert record.final_state_sha256 == _sha(2)
    assert record.evaluator_score == {
        "pass_count": 2,
        "fail_count": 0,
        "total_count": 2,
        "num_tests": 2,
        "success": True,
    }
    assert record.trace.surface_calls == (_surface_call(1), _surface_call(2))
    assert tuple(step.action for step in record.steps) == record.trace.semantic_actions
    assert tuple(step.base_call for step in record.steps) == record.trace.base_calls
    assert record.steps[0].base_observation == {"ok": (1,)}
    assert record.steps[0].surface_observation == {"wrapped": {"ok": (1,)}}


def test_execute_plan_snapshots_the_full_plan_before_the_first_callback() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}, {"ok": True}])
    adapter = _Adapter(events)
    second_call = _surface_call(2)

    def mutate_later_input() -> None:
        second_call["arguments"] = {"value": 999}

    world.requester.hook = mutate_later_input
    record = _executor(world, adapter, events).execute_plan([_surface_call(1), second_call])

    assert world.requester.calls[1]["value"] == 2
    assert record.trace.surface_calls[1]["arguments"] == {"value": 2}


def test_execute_plan_ignores_the_duck_typed_save_return_value() -> None:
    events: list[str] = []
    world = _ReturningSaveWorld(events, [{"ok": True}])

    record = _executor(world, _Adapter(events), events).execute_plan([_surface_call(1)])

    assert record.final_state_sha256 == _sha(1)
    assert events[4:7] == ["save", "hash:1", "wrap"]


def test_effect_and_transition_digests_use_structured_domain_separation() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)

    record = _executor(world, adapter, events).execute_plan([_surface_call(3)])

    before = _sha(0)
    after = _sha(1)
    call = record.trace.base_calls[0]
    effect_digest = hashlib.sha256(
        _EFFECT_DOMAIN
        + canonical_json_bytes(
            {
                "before_state_sha256": before,
                "after_state_sha256": after,
            }
        )
    ).hexdigest()
    assert record.effects[0].call_index == 0
    assert record.effects[0].base_call_fingerprint == base_call_fingerprint(call)
    assert record.effects[0].effect_digest == effect_digest
    expected_transition = hashlib.sha256(
        _TRANSITION_DOMAIN
        + canonical_json_bytes(
            {
                "effects": [
                    {
                        "call_index": 0,
                        "base_call_fingerprint": base_call_fingerprint(call),
                        "effect_digest": effect_digest,
                    }
                ],
                "final_state_sha256": after,
                "evaluator_digest": record.evaluator_digest,
            }
        )
    ).hexdigest()
    assert record.transition_digest == expected_transition


def test_transition_digest_binds_multi_step_order() -> None:
    first_events: list[str] = []
    second_events: list[str] = []
    first = _executor(
        _World(first_events, [{"ok": True}, {"ok": True}]),
        _Adapter(first_events),
        first_events,
    ).execute_plan([_surface_call(1), _surface_call(2)])
    second = _executor(
        _World(second_events, [{"ok": True}, {"ok": True}]),
        _Adapter(second_events),
        second_events,
    ).execute_plan([_surface_call(2), _surface_call(1)])

    assert first.final_state_sha256 == second.final_state_sha256
    assert first.transition_digest != second.transition_digest
    assert tuple(effect.call_index for effect in first.effects) == (0, 1)


def test_request_kwargs_and_records_are_deeply_detached_from_caller_values() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    nested = [1, {"inner": [2]}]
    call: dict[str, Any] = {
        "name": "notes__create_note",
        "arguments": {"value": nested},
    }

    record = _executor(world, adapter, events).execute_plan([call])
    nested.append(3)
    world.requester.calls[0]["value"].append(4)  # type: ignore[union-attr]

    assert record.trace.base_calls[0]["arguments"]["value"] == (1, {"inner": (2,)})
    assert record.steps[0].surface_call["arguments"]["value"] == (
        1,
        {"inner": (2,)},
    )


def test_execute_plan_returns_a_valid_unsuccessful_evaluator_record() -> None:
    events: list[str] = []
    world = _World(
        events,
        [{"error": "synthetic"}],
        tracker=_Tracker(pass_count=1, fail_count=1, num_tests=2, success=False),
    )
    record = _executor(world, _Adapter(events), events).execute_plan([_surface_call(1)])

    assert record.evaluator_score["success"] is False
    assert record.evaluator_score["fail_count"] == 1


def test_execute_plan_is_one_shot_after_success() -> None:
    events: list[str] = []
    executor = _executor(_World(events, [{"ok": True}]), _Adapter(events), events)
    executor.execute_plan([_surface_call(1)])

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan([_surface_call(2)])


class _FatalCanary(BaseException):
    pass


@pytest.mark.parametrize("error", [RuntimeError("PRIVATE_ERROR"), _FatalCanary()])
def test_execute_plan_is_one_shot_after_exception_or_baseexception(
    error: BaseException,
) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])

    def raise_error() -> None:
        raise error

    world.requester.hook = raise_error
    executor = _executor(world, _Adapter(events), events)

    if isinstance(error, Exception):
        with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$") as raised:
            executor.execute_plan([_surface_call(1)])
        assert "PRIVATE_ERROR" not in str(raised.value)
    else:
        with pytest.raises(_FatalCanary):
            executor.execute_plan([_surface_call(1)])
    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan([_surface_call(2)])


def test_baseexception_with_binding_mutation_prioritizes_integrity_error() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])

    def mutate_and_interrupt() -> None:
        world.models = object()  # type: ignore[assignment]
        raise KeyboardInterrupt

    world.requester.hook = mutate_and_interrupt
    executor = _executor(world, _Adapter(events), events)

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$"):
        executor.execute_plan([_surface_call(1)])


class _ExplodingSequence(Sequence[Mapping[str, JSONValue]]):
    def __init__(self) -> None:
        self.touches = 0

    def __getitem__(self, index: int) -> Mapping[str, JSONValue]:
        self.touches += 1
        raise AssertionError(index)

    def __len__(self) -> int:
        self.touches += 1
        raise AssertionError("sequence was touched")


class _PhaseMutatingSequence(Sequence[Mapping[str, JSONValue]]):
    def __init__(self, executor: object) -> None:
        self.executor = executor

    def __getitem__(self, index: int) -> Mapping[str, JSONValue]:
        raise IndexError(index)

    def __len__(self) -> int:
        object.__setattr__(self.executor, "_phase", "ready")
        raise RuntimeError("PRIVATE_SEQUENCE_CALLBACK")


@pytest.mark.parametrize("failure", ["empty", "compile", "save", "evaluate"])
def test_second_call_after_any_failure_does_not_touch_new_sequence(failure: str) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    executor = _executor(world, adapter, events)
    if failure == "compile":
        adapter.base_call_count = 0
    elif failure == "save":
        world.save_hook = lambda: (_ for _ in ()).throw(RuntimeError("private"))
    elif failure == "evaluate":
        world.evaluate_hook = lambda: (_ for _ in ()).throw(RuntimeError("private"))

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan([] if failure == "empty" else [_surface_call(1)])
    second_plan = _ExplodingSequence()
    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan(second_plan)
    assert second_plan.touches == 0


def test_preexisting_integrity_failure_also_consumes_the_executor() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    executor = _executor(world, adapter, events)
    adapter.surface_to_semantic = lambda call: ()  # type: ignore[method-assign]

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$"):
        executor.execute_plan([_surface_call(1)])
    del adapter.surface_to_semantic
    second_plan = _ExplodingSequence()
    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan(second_plan)
    assert second_plan.touches == 0


def test_non_method_callback_phase_mutation_preserves_integrity_priority() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    executor = _executor(world, _Adapter(events), events)

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$") as raised:
        executor.execute_plan(_PhaseMutatingSequence(executor))

    assert "PRIVATE_SEQUENCE_CALLBACK" not in str(raised.value)


def test_execute_plan_rejects_reentrancy() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    executor = _executor(world, adapter, events)

    def reenter() -> None:
        executor.execute_plan([_surface_call(9)])

    world.requester.hook = reenter
    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan([_surface_call(1)])
    assert world.save_calls == 0


@pytest.mark.parametrize("action_count", [0, 2])
def test_execute_plan_requires_one_action_per_surface_call(action_count: int) -> None:
    events: list[str] = []
    world = _World(events)
    adapter = _Adapter(events)
    adapter.action_count = action_count

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, adapter, events).execute_plan([_surface_call(1)])

    assert world.requester.calls == []
    assert world.save_calls == 0


@pytest.mark.parametrize("base_call_count", [0, 2])
def test_execute_plan_requires_one_base_call_per_action(base_call_count: int) -> None:
    events: list[str] = []
    world = _World(events)
    adapter = _Adapter(events)
    adapter.base_call_count = base_call_count

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, adapter, events).execute_plan([_surface_call(1)])

    assert world.requester.calls == []
    assert world.save_calls == 0


@pytest.mark.parametrize("calls", [[], (), "not-a-plan", {"call": "mapping"}])
def test_execute_plan_rejects_empty_or_invalid_plans_without_callbacks(calls: object) -> None:
    events: list[str] = []
    world = _World(events)

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, _Adapter(events), events).execute_plan(calls)  # type: ignore[arg-type]

    assert events == []


@pytest.mark.parametrize(
    "base_call",
    [
        {"name": "invalid", "arguments": {}},
        {"name": "notes__create_note", "arguments": {}, "metadata": "private"},
        {"name": "notes__create_note", "arguments": {"track": False}},
        {"name": "notes__create_note", "arguments": {1: "bad"}},
    ],
)
def test_execute_plan_rejects_malformed_native_calls_before_state_or_request(
    base_call: Mapping[object, object],
) -> None:
    events: list[str] = []
    world = _World(events)
    adapter = _Adapter(events)
    adapter.semantic_to_base_calls = lambda action: (base_call,)  # type: ignore[method-assign,return-value]

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$") as raised:
        _executor(world, adapter, events).execute_plan([_surface_call(1)])

    assert "private" not in str(raised.value).lower()
    assert world.requester.calls == []
    assert world.save_calls == 0
    assert not any(event.startswith("hash:") for event in events)


@pytest.mark.parametrize(
    "control_name",
    [
        "_api_name",
        "_app_name",
        "_system_datetime",
        "client",
        "raise_on_failure",
        "show",
        "track",
    ],
)
def test_execute_plan_rejects_every_requester_control_argument(
    control_name: str,
) -> None:
    events: list[str] = []
    world = _World(events)
    adapter = _Adapter(events)
    adapter.semantic_to_base_calls = lambda action: (  # type: ignore[method-assign]
        {"name": action.name, "arguments": {control_name: False}},
    )

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, adapter, events).execute_plan([_surface_call(1)])

    assert world.requester.calls == []


@pytest.mark.parametrize(
    "response",
    [None, True, 1, "text", ("tuple",), {"value": float("nan")}],
)
def test_malformed_requester_results_are_rejected_only_after_save_and_posthash(
    response: object,
) -> None:
    events: list[str] = []
    world = _World(events, [response])

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, _Adapter(events), events).execute_plan([_surface_call(1)])

    assert events[:6] == ["parse", "compile", "hash:0", "request", "save", "hash:1"]
    assert "wrap" not in events
    assert "evaluate" not in events


class _DictSubclass(dict[str, object]):
    pass


class _ListSubclass(list[object]):
    pass


@pytest.mark.parametrize("response", [_DictSubclass(ok=True), _ListSubclass([1])])
def test_requester_result_root_must_be_an_exact_builtin_container(
    response: object,
) -> None:
    events: list[str] = []
    world = _World(events, [response])

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, _Adapter(events), events).execute_plan([_surface_call(1)])

    assert world.save_calls == 1
    assert events[-1] == "hash:1"


def test_cyclic_requester_result_is_rejected_after_save_and_posthash() -> None:
    events: list[str] = []
    response: dict[str, object] = {}
    response["cycle"] = response
    world = _World(events, [response])

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, _Adapter(events), events).execute_plan([_surface_call(1)])

    assert world.save_calls == 1
    assert events[-1] == "hash:1"


def test_execute_plan_detects_interstep_state_drift_before_second_request() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}, {"ok": True}])
    adapter = _Adapter(events)
    wrap_calls = 0

    def drift_after_first_wrap() -> None:
        nonlocal wrap_calls
        wrap_calls += 1
        if wrap_calls == 1:
            world.models.state += 10

    adapter.wrap_hook = drift_after_first_wrap

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, adapter, events).execute_plan([_surface_call(1), _surface_call(2)])

    assert len(world.requester.calls) == 1
    assert world.save_calls == 1
    assert events[-1] == "hash:11"


@pytest.mark.parametrize("phase", ["parse", "compile", "wrap", "canonicalize"])
def test_adapter_cannot_mutate_objects_that_feed_the_final_record(phase: str) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _InputMutatingAdapter(events, phase)

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$") as raised:
        _executor(world, adapter, events).execute_plan([_surface_call(1)])

    assert "PRIVATE" not in str(raised.value)


def test_rollout_uses_the_attested_adapter_callback_not_a_dynamic_reread() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _OneReadForgingAdapter(events)
    executor = _executor(world, adapter, events)
    adapter.armed = True

    record = executor.execute_plan([_surface_call(1)])

    assert "forged-parse" not in events
    assert record.trace.semantic_actions == (SemanticAction("notes__create_note", {"value": 1}),)


@pytest.mark.parametrize("phase", ["request", "save", "evaluate"])
def test_execute_plan_detects_world_binding_mutation_with_integrity_priority(
    phase: str,
) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)

    def mutate() -> None:
        world.requester = Any  # type: ignore[assignment]

    if phase == "request":
        world.requester.hook = mutate
    elif phase == "save":
        world.save_hook = mutate
    else:
        world.evaluate_hook = mutate

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$"):
        _executor(world, adapter, events).execute_plan([_surface_call(1)])


def test_callback_phase_mutation_is_an_integrity_failure() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    executor = _executor(world, _Adapter(events), events)

    def mutate_phase() -> None:
        object.__setattr__(executor, "_phase", "consumed")

    world.requester.hook = mutate_phase

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$"):
        executor.execute_plan([_surface_call(1)])

    assert world.save_calls == 0


@pytest.mark.parametrize("target", ["request-seal", "adapter-method-seals"])
def test_equal_replacement_of_executor_seal_roots_is_rejected(target: str) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    executor = _executor(world, _Adapter(events), events)

    if target == "request-seal":
        seal = object.__getattribute__(executor, "_request_seal")
        replacement = type(seal)(seal.callback, seal.bound_self, seal.bound_function)
        assert replacement == seal and replacement is not seal
        object.__setattr__(executor, "_request_seal", replacement)
    else:
        seals = object.__getattribute__(executor, "_adapter_method_seals")
        replacement = tuple(list(seals))
        assert replacement == seals and replacement is not seals
        object.__setattr__(executor, "_adapter_method_seals", replacement)

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$"):
        executor.execute_plan([_surface_call(1)])


def test_in_place_callable_seal_callback_tampering_is_rejected() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    executor = _executor(world, _Adapter(events), events)
    seal = object.__getattribute__(executor, "_request_seal")
    object.__setattr__(seal, "callback", lambda **kwargs: {"private": kwargs})

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$"):
        executor.execute_plan([_surface_call(1)])

    assert world.requester.calls == []


@pytest.mark.parametrize(
    "target",
    [
        "request",
        "save",
        "evaluate",
        "parse",
        "compile",
        "wrap",
        "canonicalize",
        "variant",
        "models",
        "state-hasher",
        "evaluator-hasher",
    ],
)
def test_preexisting_executor_binding_tampering_is_rejected(target: str) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    executor = _executor(world, adapter, events)
    if target == "request":
        world.requester.request = lambda **kwargs: {"ok": kwargs}  # type: ignore[method-assign]
    elif target == "save":
        world.save = lambda: None  # type: ignore[method-assign]
    elif target == "evaluate":
        world.evaluate = lambda **kwargs: world.tracker  # type: ignore[method-assign]
    elif target == "parse":
        adapter.surface_to_semantic = lambda call: ()  # type: ignore[method-assign]
    elif target == "compile":
        adapter.semantic_to_base_calls = lambda action: ()  # type: ignore[method-assign]
    elif target == "wrap":
        adapter.base_observation_to_surface = lambda *args: None  # type: ignore[method-assign]
    elif target == "canonicalize":
        adapter.canonicalize_trace = lambda trace: ()  # type: ignore[method-assign]
    elif target == "variant":
        object.__setattr__(adapter, "_variant", object())
    elif target == "models":
        world.models = _Models()
    elif target == "state-hasher":
        object.__setattr__(executor, "_state_hasher", lambda models: _sha(models))
    else:
        object.__setattr__(executor, "_evaluator_hasher", lambda tracker: _sha(tracker))

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$"):
        executor.execute_plan([_surface_call(1)])


def test_different_bound_method_receiver_cannot_spoof_a_binding() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    executor = _executor(world, _Adapter(events), events)
    other_world = _World([], [{"ok": True}])
    world.save = _World.save.__get__(other_world, _World)  # type: ignore[method-assign]

    with pytest.raises(ValueError, match=f"^{_INTEGRITY_ERROR}$"):
        executor.execute_plan([_surface_call(1)])


@pytest.mark.parametrize(
    ("phase", "expected_events"),
    [
        ("parse", ["parse"]),
        ("compile", ["parse", "compile"]),
        ("prehash", ["parse", "compile", "hash:0"]),
        ("request", ["parse", "compile", "hash:0", "request"]),
        ("save", ["parse", "compile", "hash:0", "request", "save"]),
        (
            "posthash",
            ["parse", "compile", "hash:0", "request", "save", "hash:1"],
        ),
        (
            "wrap",
            ["parse", "compile", "hash:0", "request", "save", "hash:1", "wrap"],
        ),
        (
            "canonicalize",
            [
                "parse",
                "compile",
                "hash:0",
                "request",
                "save",
                "hash:1",
                "wrap",
                "canonicalize",
            ],
        ),
        (
            "evaluate",
            [
                "parse",
                "compile",
                "hash:0",
                "request",
                "save",
                "hash:1",
                "wrap",
                "canonicalize",
                "evaluate",
            ],
        ),
        (
            "evaluator-hash",
            [
                "parse",
                "compile",
                "hash:0",
                "request",
                "save",
                "hash:1",
                "wrap",
                "canonicalize",
                "evaluate",
                "evaluator-hash",
            ],
        ),
    ],
)
def test_all_callback_failures_are_sanitized(
    phase: str,
    expected_events: list[str],
) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    private_error = RuntimeError("PRIVATE_CALLBACK_PAYLOAD /private/path")

    def explode() -> None:
        raise private_error

    state_calls = 0

    def state_hasher(models: object) -> str:
        nonlocal state_calls
        state_calls += 1
        events.append(f"hash:{getattr(models, 'state', -1)}")
        if phase == "prehash" and state_calls == 1:
            raise private_error
        if phase == "posthash" and state_calls == 2:
            raise private_error
        return _sha(getattr(models, "state", None))

    if phase == "parse":
        adapter.parse_hook = explode
    elif phase == "compile":
        adapter.compile_hook = explode
    elif phase == "request":
        world.requester.hook = explode
    elif phase == "save":
        world.save_hook = explode
    elif phase == "wrap":
        adapter.wrap_hook = explode
    elif phase == "canonicalize":
        adapter.canonicalize_hook = explode
    elif phase == "evaluate":
        world.evaluate_hook = explode

    def evaluator_hasher(tracker: object) -> str:
        events.append("evaluator-hash")
        if phase == "evaluator-hash":
            raise private_error
        return _sha(tracker)

    from toolshift.benchmarks.appworld_runtime import AppWorldEpisodeExecutor

    executor = AppWorldEpisodeExecutor(
        world,
        adapter,
        state_hasher=state_hasher,
        evaluator_hasher=evaluator_hasher,
    )
    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$") as raised:
        executor.execute_plan([_surface_call(1)])

    assert "PRIVATE_CALLBACK_PAYLOAD" not in str(raised.value)
    assert "/private/path" not in repr(raised.value)
    assert events == expected_events


@pytest.mark.parametrize(
    "tracker",
    [
        _Tracker(pass_count=True),
        _Tracker(pass_count=-1, fail_count=1),
        _Tracker(pass_count=2, fail_count=1, num_tests=2),
    ],
)
def test_execute_plan_rejects_malformed_compact_evaluator_scores(
    tracker: _Tracker,
) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}], tracker=tracker)

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, _Adapter(events), events).execute_plan([_surface_call(1)])


def test_execute_plan_rejects_invalid_hasher_digests() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    from toolshift.benchmarks.appworld_runtime import AppWorldEpisodeExecutor

    executor = AppWorldEpisodeExecutor(
        world,
        adapter,
        state_hasher=lambda models: "not-a-digest",
        evaluator_hasher=lambda tracker: "also-not-a-digest",
    )

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan([_surface_call(1)])
    assert world.requester.calls == []


def test_execute_plan_rejects_an_invalid_evaluator_digest_independently() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    from toolshift.benchmarks.appworld_runtime import AppWorldEpisodeExecutor

    executor = AppWorldEpisodeExecutor(
        world,
        adapter,
        state_hasher=_state_hasher(events),
        evaluator_hasher=lambda tracker: "not-a-digest",
    )

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan([_surface_call(1)])
    assert world.save_calls == 1
    assert world.evaluate_calls == [False]


@pytest.mark.parametrize("result_kind", ["list", "wrong-action"])
def test_execute_plan_rejects_malformed_canonicalization_results(
    result_kind: str,
) -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    if result_kind == "list":
        adapter.canonicalize_trace = lambda trace: list(trace.semantic_actions)  # type: ignore[method-assign,return-value]
    else:
        adapter.canonicalize_trace = lambda trace: (  # type: ignore[method-assign]
            SemanticAction("notes__create_note", {"value": 999}),
        )

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        _executor(world, adapter, events).execute_plan([_surface_call(1)])


def test_evaluator_hasher_cannot_change_the_score_snapshot() -> None:
    events: list[str] = []
    world = _World(events, [{"ok": True}])
    adapter = _Adapter(events)
    from toolshift.benchmarks.appworld_runtime import AppWorldEpisodeExecutor

    def mutating_hasher(tracker: object) -> str:
        assert type(tracker) is _Tracker
        tracker.pass_count = 1
        tracker.fail_count = 1
        tracker.total_count = 2
        tracker.success = False
        return _sha("changed")

    executor = AppWorldEpisodeExecutor(
        world,
        adapter,
        state_hasher=_state_hasher(events),
        evaluator_hasher=mutating_hasher,
    )

    with pytest.raises(ValueError, match=f"^{_EXECUTION_ERROR}$"):
        executor.execute_plan([_surface_call(1)])


def test_executor_exposes_no_reset_load_or_code_execution_api() -> None:
    events: list[str] = []
    executor = _executor(_World(events), _Adapter(events), events)

    for name in ("reset", "load_state", "execute", "execute_code", "close"):
        assert not hasattr(executor, name)


def test_private_records_and_executor_repr_do_not_expose_episode_payloads() -> None:
    events: list[str] = []
    world = _World(events, [{"PRIVATE_OBSERVATION_CANARY": True}])
    adapter = _Adapter(events)
    executor = _executor(world, adapter, events)
    record = executor.execute_plan([_surface_call(987654)])

    for rendered in (repr(executor), repr(record), repr(record.steps[0])):
        assert "987654" not in rendered
        assert "PRIVATE_OBSERVATION_CANARY" not in rendered
        assert "notes__create_note" not in rendered


def test_episode_record_trace_score_and_steps_are_deeply_immutable() -> None:
    events: list[str] = []
    record = _executor(
        _World(events, [{"ok": [1]}]),
        _Adapter(events),
        events,
    ).execute_plan([_surface_call(1)])

    with pytest.raises((AttributeError, TypeError)):
        record.steps += record.steps
    with pytest.raises((AttributeError, TypeError)):
        record.evaluator_score["success"] = False  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        record.trace.surface_calls[0]["arguments"]["value"] = 2  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        record.steps[0].base_observation["ok"].append(2)  # type: ignore[index,union-attr]
