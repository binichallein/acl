"""Unit tests for the privacy-preserving AppWorld replay verifier."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

import toolshift.benchmarks.appworld_replay as replay_module
from toolshift.benchmarks.appworld_replay import (
    PINNED_APPWORLD_COMMIT,
    ReplayMode,
    ReplayRun,
    ReplayStage,
    TaskReplayResult,
    TaskSet,
    build_task_set,
    canonical_state_sha256,
    evaluator_sha256,
    exit_code_for_summary,
    require_pinned_appworld_revision,
    run_replay_verification,
    run_task_repetitions,
    summarize_replays,
    write_summary_json,
)
from toolshift.types import manifest_sha256


class _FakeSQLModel:
    def __init__(self, model_names: list[str]) -> None:
        self._model_names = model_names

    def model_names(self) -> list[str]:
        return list(self._model_names)


class _FakeAppModels:
    def __init__(self, model_names: list[str]) -> None:
        self.SQLModel = _FakeSQLModel(model_names)


class _FakeModels(dict[str, _FakeAppModels]):
    def __init__(
        self,
        apps: dict[str, list[str]],
        rows: dict[tuple[str, str], list[tuple[int, str]]],
    ) -> None:
        super().__init__(
            (app_name, _FakeAppModels(model_names))
            for app_name, model_names in apps.items()
        )
        self._rows = rows
        self.clear_calls = 0
        self.lookups: list[tuple[str, str]] = []

    def clear_ids_record_hashes(self) -> None:
        self.clear_calls += 1

    def ids_record_hashes(self, app_name: str, model_name: str) -> list[tuple[int, str]]:
        assert self.clear_calls > 0, "cached record hashes must be cleared before lookup"
        self.lookups.append((app_name, model_name))
        return list(self._rows[(app_name, model_name)])


def _models(
    *,
    reverse: bool = False,
    changed_hash: str = "hash-b",
) -> _FakeModels:
    apps = {
        "calendar": ["Event", "AuditModelHash", "Empty"],
        "contacts": ["Contact"],
    }
    rows = {
        ("calendar", "Event"): [(2, changed_hash), (1, "hash-a")],
        ("calendar", "Empty"): [],
        ("calendar", "AuditModelHash"): [(9, "ignored")],
        ("contacts", "Contact"): [(3, "hash-c")],
    }
    if reverse:
        apps = {
            app_name: list(reversed(model_names))
            for app_name, model_names in reversed(tuple(apps.items()))
        }
        rows = {key: list(reversed(value)) for key, value in reversed(tuple(rows.items()))}
    return _FakeModels(apps, rows)


def test_state_digest_is_order_stable_clears_cache_and_excludes_model_hash_tables() -> None:
    first = _models()
    reordered = _models(reverse=True)

    first_digest = canonical_state_sha256(first)
    reordered_digest = canonical_state_sha256(reordered)

    assert first_digest == reordered_digest
    assert first.clear_calls == 1
    assert reordered.clear_calls == 1
    assert first.lookups == [
        ("calendar", "Empty"),
        ("calendar", "Event"),
        ("contacts", "Contact"),
    ]
    assert all(not model_name.endswith("ModelHash") for _, model_name in first.lookups)


def test_state_digest_changes_for_record_content_but_not_excluded_metadata() -> None:
    baseline = _models()
    changed = _models(changed_hash="different")
    metadata_only = _models()
    metadata_only._rows[("calendar", "AuditModelHash")] = [(9, "different")]

    assert canonical_state_sha256(baseline) != canonical_state_sha256(changed)
    assert canonical_state_sha256(_models()) == canonical_state_sha256(metadata_only)


def test_state_projection_sorts_records_by_native_id_then_record_hash() -> None:
    models = _FakeModels(
        {"notes": ["Note"]},
        {("notes", "Note"): [(10, "hash-z"), (2, "hash-a")]},
    )
    expected = manifest_sha256(
        [
            {
                "app": "notes",
                "models": [
                    {
                        "model": "Note",
                        "records": [(2, "hash-a"), (10, "hash-z")],
                    }
                ],
            }
        ]
    )

    assert canonical_state_sha256(models) == expected


@pytest.mark.parametrize(
    "row",
    [("1", "hash-a"), (True, "hash-a"), (1, ""), (1, " \t\n")],
    ids=["string-id", "boolean-id", "empty-record-hash", "blank-record-hash"],
)
def test_state_projection_fails_closed_on_pinned_record_hash_contract_drift(
    row: tuple[object, object],
) -> None:
    models = _FakeModels(
        {"notes": ["Note"]},
        {("notes", "Note"): [row]},  # type: ignore[list-item]
    )

    with pytest.raises(ValueError, match="record"):
        canonical_state_sha256(models)


class _FakeTracker:
    def __init__(self, *, pass_count: int = 2, no_op_pass_failures: int = 0) -> None:
        self.pass_count = pass_count
        self.fail_count = 3 - pass_count
        self.total_count = 3
        self.num_tests = 3
        self.pass_percentage = 100 * pass_count / 3
        self.success = pass_count == 3
        self.difficulty = 2
        self.failures = [
            {"requirement": "private", "trace": "private", "label": "no_op_pass"}
            for _ in range(no_op_pass_failures)
        ]

    def to_dict(self, *, stats_only: bool) -> dict[str, object]:
        assert stats_only is True
        return {
            "success": self.success,
            "difficulty": self.difficulty,
            "num_tests": self.num_tests,
        }


def test_evaluator_digest_covers_score_fields_and_only_counts_no_op_failures() -> None:
    baseline = evaluator_sha256(_FakeTracker(pass_count=2, no_op_pass_failures=0))
    changed_score = evaluator_sha256(_FakeTracker(pass_count=1, no_op_pass_failures=0))
    changed_no_op = evaluator_sha256(_FakeTracker(pass_count=2, no_op_pass_failures=1))

    assert baseline != changed_score
    assert baseline != changed_no_op


class _FakeRequester:
    def __init__(self, requests: object) -> None:
        self.requests = requests


def _valid_request_trace() -> list[dict[str, object]]:
    return [
        {
            "method": "POST",
            "url": "/notes",
            "data": {"title": "unit-test"},
        }
    ]


class _FakeWorld:
    def __init__(
        self,
        serial: int,
        *,
        execution_output: object = "ok",
        request_trace: object | None = None,
    ) -> None:
        self.serial = serial
        self.closed = False
        self.execution_output = execution_output
        self.request_trace_after_execute = (
            _valid_request_trace() if request_trace is None else request_trace
        )
        self.models = _FakeModels(
            {"notes": ["Note", "NoteModelHash"]},
            {
                ("notes", "Note"): [(1, "initial")],
                ("notes", "NoteModelHash"): [(1, "cache")],
            },
        )
        self.requester = _FakeRequester([])
        self.task = SimpleNamespace(
            ground_truth=SimpleNamespace(
                compiled_solution_code="def solution(apis, requester): return None"
            )
        )
        self.executed_code: str | None = None
        self.evaluate_calls: list[bool] = []

    def execute(self, code: str) -> object:
        self.executed_code = code
        if self.execution_output != "ok":
            return self.execution_output
        self.models._rows[("notes", "Note")] = [(1, "terminal")]
        self.requester.requests = self.request_trace_after_execute
        return "ok"

    def evaluate(self, *, suppress_errors: bool) -> _FakeTracker:
        self.evaluate_calls.append(suppress_errors)
        return _FakeTracker(pass_count=3)


class _WorldContext(AbstractContextManager[_FakeWorld]):
    def __init__(self, world: _FakeWorld) -> None:
        self.world = world

    def __enter__(self) -> _FakeWorld:
        return self.world

    def __exit__(self, *exc_info: object) -> None:
        self.world.closed = True


class _RecordingFactory:
    def __init__(
        self,
        *,
        execution_outputs: dict[int, object] | None = None,
        request_traces: dict[int, object] | None = None,
        raise_run: int | None = None,
    ) -> None:
        self.execution_outputs = execution_outputs or {}
        self.request_traces = request_traces or {}
        self.raise_run = raise_run
        self.calls: list[dict[str, object]] = []
        self.worlds: list[_FakeWorld] = []

    def __call__(self, **kwargs: object) -> _WorldContext:
        serial = len(self.calls)
        self.calls.append(dict(kwargs))
        if serial == self.raise_run:
            raise RuntimeError("sensitive exception payload")
        world = _FakeWorld(
            serial,
            execution_output=self.execution_outputs.get(serial, "ok"),
            request_trace=self.request_traces.get(serial),
        )
        self.worlds.append(world)
        return _WorldContext(world)


def _lifecycle_appworld_class(
    *,
    constructor_error: Exception | None = None,
    close_error: Exception | None = None,
    close_all_error: Exception | None = None,
) -> type[object]:
    class LifecycleAppWorld:
        instances: ClassVar[list[LifecycleAppWorld]] = []
        constructor_calls = 0
        close_all_calls = 0

        def __init__(self, **kwargs: object) -> None:
            type(self).constructor_calls += 1
            self.kwargs = kwargs
            self.close_calls = 0
            if constructor_error is not None:
                raise constructor_error
            type(self).instances.append(self)

        def close(self) -> None:
            self.close_calls += 1
            if close_error is not None:
                raise close_error

        @classmethod
        def close_all(cls) -> None:
            cls.close_all_calls += 1
            if close_all_error is not None:
                raise close_all_error

    return LifecycleAppWorld


def test_managed_appworld_context_closes_once_after_normal_body() -> None:
    appworld_class = _lifecycle_appworld_class()

    with replay_module.managed_appworld_context(
        appworld_class,
        task_id="train-task-unit",
    ) as world:
        assert world.kwargs == {"task_id": "train-task-unit"}

    assert world.close_calls == 1
    assert appworld_class.close_all_calls == 0


def test_managed_appworld_context_closes_once_after_body_exception() -> None:
    appworld_class = _lifecycle_appworld_class()

    with (
        pytest.raises(LookupError, match="body failed"),
        replay_module.managed_appworld_context(appworld_class),
    ):
        raise LookupError("body failed")

    assert appworld_class.instances[0].close_calls == 1
    assert appworld_class.close_all_calls == 0


def test_managed_appworld_context_uses_close_all_after_constructor_failure() -> None:
    appworld_class = _lifecycle_appworld_class(
        constructor_error=LookupError("constructor failed")
    )

    with (
        pytest.raises(LookupError, match="constructor failed"),
        replay_module.managed_appworld_context(appworld_class),
    ):
        pytest.fail("constructor failure must prevent context entry")

    assert appworld_class.constructor_calls == 1
    assert appworld_class.close_all_calls == 1


def test_managed_appworld_context_falls_back_after_close_failure_without_double_close() -> None:
    appworld_class = _lifecycle_appworld_class(close_error=LookupError("close failed"))

    with (
        pytest.raises(LookupError, match="close failed"),
        replay_module.managed_appworld_context(appworld_class),
    ):
        pass

    assert appworld_class.instances[0].close_calls == 1
    assert appworld_class.close_all_calls == 1


def test_managed_appworld_context_surfaces_failed_close_all_fallback() -> None:
    appworld_class = _lifecycle_appworld_class(
        close_error=LookupError("close failed"),
        close_all_error=RuntimeError("close_all failed"),
    )

    with (
        pytest.raises(RuntimeError, match="global cleanup failed") as error,
        replay_module.managed_appworld_context(appworld_class),
    ):
        pass

    assert type(error.value.__cause__) is RuntimeError
    assert type(error.value.__cause__.__context__) is LookupError
    assert appworld_class.instances[0].close_calls == 1
    assert appworld_class.close_all_calls == 1


def test_repetitions_use_fresh_worlds_exact_runtime_options_and_context_cleanup() -> None:
    factory = _RecordingFactory()

    result = run_task_repetitions(
        "train-task-unit",
        seed=100,
        repetitions=3,
        world_context_factory=factory,
    )

    assert len(factory.worlds) == 3
    assert len({id(world) for world in factory.worlds}) == 3
    assert len({call["experiment_name"] for call in factory.calls}) == 3
    assert all(
        {
            key: value
            for key, value in call.items()
            if key != "experiment_name"
        }
        == {
            "task_id": "train-task-unit",
            "ground_truth_mode": "full",
            "raise_on_failure": False,
            "random_seed": 100,
        }
        for call in factory.calls
    )
    assert all(world.closed for world in factory.worlds)
    assert all(world.evaluate_calls == [False] for world in factory.worlds)
    assert all(
        world.executed_code is not None
        and world.executed_code.endswith("\nsolution(apis, requester)")
        for world in factory.worlds
    )
    assert result.initial_state_matches is True
    assert result.final_state_matches is True
    assert result.evaluator_matches is True
    assert result.trace_matches is True
    assert result.all_oracles_succeeded is True


def test_sequential_verification_aggregates_without_returning_task_ids() -> None:
    factory = _RecordingFactory()
    task_set = build_task_set({"train": ["train-task-unit"]})

    summary = run_replay_verification(
        task_set,
        mode=ReplayMode.SMOKE,
        seed=100,
        repetitions=2,
        workers=1,
        world_context_factory=factory,
    )

    assert summary.task_count == 1
    assert summary.episode_count == 2
    assert summary.smoke_passed is True
    assert "train-task-unit" not in json.dumps(summary.to_dict())


def test_traceback_and_exception_fail_closed_without_exposing_payloads() -> None:
    traceback_result = run_task_repetitions(
        "train-task-unit",
        seed=100,
        repetitions=3,
        world_context_factory=_RecordingFactory(
            execution_outputs={
                1: "Execution failed. Traceback:\nTimeoutError: sensitive payload"
            }
        ),
    )
    exception_result = run_task_repetitions(
        "train-task-unit",
        seed=100,
        repetitions=3,
        world_context_factory=_RecordingFactory(raise_run=1),
    )

    assert traceback_result.execution_failure_count == 1
    assert traceback_result.runs[1].failure_stage == ReplayStage.EXECUTE_ORACLE
    assert traceback_result.runs[1].exception_type is None
    assert exception_result.execution_failure_count == 1
    assert exception_result.exception_count == 1
    assert exception_result.runs[1].failure_stage == ReplayStage.CREATE_WORLD
    assert exception_result.runs[1].exception_type == "RuntimeError"
    assert "sensitive" not in repr(exception_result)
    assert exception_result.final_state_matches is False


@pytest.mark.parametrize(
    "unsafe_output",
    [{"unexpected": "mapping"}, ["unexpected", "list"], None],
    ids=["mapping", "list", "none"],
)
def test_non_string_execute_output_fails_closed_as_unsafe_shape(
    unsafe_output: object,
) -> None:
    factory = _RecordingFactory(execution_outputs={1: unsafe_output})

    result = run_task_repetitions(
        "train-task-unit",
        seed=100,
        repetitions=3,
        world_context_factory=factory,
    )

    failed_run = result.runs[1]
    assert failed_run.execution_failed is True
    assert failed_run.failure_stage is ReplayStage.EXECUTE_ORACLE
    assert failed_run.exception_type == "ValueError"
    assert factory.worlds[1].evaluate_calls == []


@pytest.mark.parametrize(
    "unsafe_trace",
    [
        [],
        "not-a-list",
        ["not-a-request"],
        [{"method": "", "url": "/notes", "data": {}}],
        [{"method": "POST", "url": "  ", "data": {}}],
        [{"method": "POST", "url": "/notes", "data": []}],
        [{"method": "POST", "url": "/notes"}],
    ],
    ids=[
        "empty",
        "non-list",
        "non-mapping-item",
        "blank-method",
        "blank-url",
        "non-mapping-data",
        "missing-data",
    ],
)
def test_native_request_trace_must_be_nonempty_and_match_pinned_shape(
    unsafe_trace: object,
) -> None:
    factory = _RecordingFactory(request_traces={1: unsafe_trace})

    result = run_task_repetitions(
        "train-task-unit",
        seed=100,
        repetitions=3,
        world_context_factory=factory,
    )

    failed_run = result.runs[1]
    assert failed_run.execution_failed is True
    assert failed_run.failure_stage is ReplayStage.TRACE_FINGERPRINT
    assert failed_run.exception_type == "ValueError"
    assert factory.worlds[1].evaluate_calls == []
    assert result.is_consistent is False


def _run(
    *,
    initial: str = "initial",
    final: str = "final",
    evaluator: str = "evaluator",
    trace: str = "trace",
    success: bool = True,
    execution_failed: bool = False,
    failure_stage: ReplayStage | None = None,
    exception_type: str | None = None,
) -> ReplayRun:
    return ReplayRun(
        initial_state_sha256=initial,
        final_state_sha256=final,
        evaluator_sha256=evaluator,
        trace_sha256=trace,
        oracle_success=success,
        execution_failed=execution_failed,
        failure_stage=failure_stage,
        exception_type=exception_type,
    )


def test_failed_replay_run_cannot_claim_oracle_success() -> None:
    with pytest.raises(ValueError, match="oracle_success"):
        _run(
            success=True,
            execution_failed=True,
            failure_stage=ReplayStage.EXECUTE_ORACLE,
        )


def _task_result(*runs: ReplayRun) -> TaskReplayResult:
    return TaskReplayResult(runs=tuple(runs))


def test_task_replay_result_takes_an_immutable_snapshot_of_runs() -> None:
    mutable_runs = [_run(), _run()]

    result = TaskReplayResult(runs=mutable_runs)  # type: ignore[arg-type]
    mutable_runs.append(_run(final="changed"))

    assert isinstance(result.runs, tuple)
    assert len(result.runs) == 2
    with pytest.raises(FrozenInstanceError):
        result.runs = ()  # type: ignore[misc]


def test_summary_rejects_non_task_results_and_forged_run_shapes() -> None:
    task_set = build_task_set({"train": ["train-task-unit"]})

    with pytest.raises(ValueError, match="TaskReplayResult"):
        summarize_replays(
            task_set,
            [object()],  # type: ignore[list-item]
            mode=ReplayMode.SMOKE,
            seed=100,
            repetitions=2,
            workers=1,
        )

    forged_result = _task_result(_run(), _run())
    object.__setattr__(forged_result.runs[0], "execution_failed", True)
    with pytest.raises(ValueError, match="replay result integrity"):
        summarize_replays(
            task_set,
            [forged_result],
            mode=ReplayMode.SMOKE,
            seed=100,
            repetitions=2,
            workers=1,
        )


def _official_task_set() -> object:
    return build_task_set(
        {
            "train": [f"train-{index:03d}" for index in range(90)],
            "dev": [f"dev-{index:03d}" for index in range(57)],
        }
    )


def test_task_set_rejects_held_out_empty_duplicate_and_overlapping_ids() -> None:
    with pytest.raises(ValueError, match="allowed"):
        build_task_set({"test_normal": ["hidden"]})
    with pytest.raises(ValueError, match="non-empty"):
        build_task_set({"train": []})
    with pytest.raises(ValueError, match="non-blank"):
        build_task_set({"train": [""]})
    with pytest.raises(ValueError, match="unique"):
        build_task_set({"train": ["same", "same"]})
    with pytest.raises(ValueError, match="unique"):
        build_task_set({"train": ["same"], "dev": ["same"]})


def test_direct_task_set_construction_derives_a_deep_immutable_snapshot() -> None:
    train_ids = ["train-b", "train-a"]
    source = {"train": train_ids}

    task_set = TaskSet(source)
    train_ids.append("train-c")
    source["dev"] = ["dev-a"]

    assert task_set.task_ids == ("train-a", "train-b")
    assert task_set.split_counts == (("train", 2),)
    assert tuple(task_set.task_ids_by_split["train"]) == ("train-a", "train-b")
    with pytest.raises(TypeError):
        task_set.task_ids_by_split["dev"] = ("dev-a",)  # type: ignore[index]


def test_direct_task_set_construction_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        TaskSet({"train": ["same"], "dev": ["same"]})


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("task_ids", ("forged",)),
        ("split_counts", (("dev", 57), ("train", 90))),
        ("task_set_sha256", "0" * 64),
    ],
    ids=["task-ids", "split-counts", "fingerprint"],
)
def test_summary_rejects_low_level_task_set_forgery(
    field_name: str,
    forged_value: object,
) -> None:
    task_set = TaskSet({"train": ["train-task-unit"]})
    object.__setattr__(task_set, field_name, forged_value)

    with pytest.raises(ValueError, match="integrity"):
        summarize_replays(
            task_set,
            [_task_result(_run(), _run())],
            mode=ReplayMode.SMOKE,
            seed=100,
            repetitions=2,
            workers=1,
        )


def test_task_set_is_sorted_and_fingerprint_is_order_stable() -> None:
    first = build_task_set({"dev": ["d-2", "d-1"], "train": ["t-1"]})
    second = build_task_set({"train": ["t-1"], "dev": ["d-1", "d-2"]})

    assert first.task_ids == ("d-1", "d-2", "t-1")
    assert first.task_set_sha256 == second.task_set_sha256


def test_formal_summary_passes_only_for_complete_fixed_official_protocol() -> None:
    task_set = _official_task_set()
    result = _task_result(_run(), _run(), _run())

    summary = summarize_replays(
        task_set,
        [result] * 147,
        mode=ReplayMode.GATE,
        seed=100,
        repetitions=3,
        workers=1,
    )

    assert summary.gate_evaluable is True
    assert summary.gate_passed is True
    assert summary.initial_state_match_rate == 1.0
    assert summary.final_state_match_rate == 1.0
    assert summary.evaluator_match_rate == 1.0
    assert summary.trace_match_rate == 1.0
    assert summary.task_consistency_rate == 1.0
    assert summary.oracle_success_rate == 1.0
    assert summary.execution_failure_count == 0
    assert summary.exception_count == 0


@pytest.mark.parametrize(
    ("task_set", "seed", "repetitions", "workers"),
    [
        (build_task_set({"train": [f"t-{index}" for index in range(99)]}), 100, 3, 1),
        (_official_task_set(), 101, 3, 1),
        (_official_task_set(), 100, 2, 1),
        (_official_task_set(), 100, 3, 2),
    ],
)
def test_incomplete_or_nonstandard_protocol_cannot_report_gate_passed(
    task_set: object,
    seed: int,
    repetitions: int,
    workers: int,
) -> None:
    run = _run()
    results = [_task_result(*(run for _ in range(repetitions)))] * len(task_set.task_ids)

    summary = summarize_replays(
        task_set,
        results,
        mode=ReplayMode.GATE,
        seed=seed,
        repetitions=repetitions,
        workers=workers,
    )

    assert summary.gate_evaluable is False
    assert summary.gate_passed is False


@pytest.mark.parametrize(
    "task_set",
    [
        build_task_set({"train": [f"t-{index}" for index in range(147)]}),
        build_task_set(
            {
                "train": [f"t-{index}" for index in range(89)],
                "dev": [f"d-{index}" for index in range(58)],
            }
        ),
    ],
    ids=["missing-dev-split", "wrong-split-counts"],
)
def test_147_tasks_with_nonofficial_split_composition_is_not_gate_evaluable(
    task_set: object,
) -> None:
    result = _task_result(_run(), _run(), _run())

    summary = summarize_replays(
        task_set,
        [result] * 147,
        mode=ReplayMode.GATE,
        seed=100,
        repetitions=3,
        workers=1,
    )

    assert summary.gate_evaluable is False
    assert summary.gate_passed is False


def test_one_mismatch_of_147_passes_point_99_but_two_mismatches_fail() -> None:
    task_set = _official_task_set()
    matched = _task_result(_run(), _run(), _run())
    mismatched = _task_result(_run(), _run(final="changed"), _run())

    one = summarize_replays(
        task_set,
        [mismatched, *([matched] * 146)],
        mode=ReplayMode.GATE,
        seed=100,
        repetitions=3,
        workers=1,
    )
    two = summarize_replays(
        task_set,
        [mismatched, mismatched, *([matched] * 145)],
        mode=ReplayMode.GATE,
        seed=100,
        repetitions=3,
        workers=1,
    )

    assert one.final_state_match_rate == pytest.approx(146 / 147)
    assert one.gate_passed is True
    assert two.final_state_match_rate == pytest.approx(145 / 147)
    assert two.gate_passed is False


def test_joint_task_consistency_rejects_disjoint_channel_mismatches() -> None:
    task_set = _official_task_set()
    matched = _task_result(_run(), _run(), _run())
    final_mismatch = _task_result(_run(), _run(final="changed"), _run())
    evaluator_mismatch = _task_result(
        _run(),
        _run(evaluator="changed"),
        _run(),
    )
    trace_mismatch = _task_result(_run(), _run(trace="changed"), _run())

    summary = summarize_replays(
        task_set,
        [final_mismatch, evaluator_mismatch, trace_mismatch, *([matched] * 144)],
        mode=ReplayMode.GATE,
        seed=100,
        repetitions=3,
        workers=1,
    )

    assert summary.final_state_match_rate == pytest.approx(146 / 147)
    assert summary.evaluator_match_rate == pytest.approx(146 / 147)
    assert summary.trace_match_rate == pytest.approx(146 / 147)
    assert summary.task_consistency_rate == pytest.approx(144 / 147)
    assert summary.gate_passed is False


def test_exception_and_initial_state_mismatch_always_fail_gate() -> None:
    task_set = _official_task_set()
    matched = _task_result(_run(), _run(), _run())
    failed = _task_result(
        _run(),
        _run(
            initial="changed",
            final="",
            evaluator="",
            trace="",
            success=False,
            execution_failed=True,
            failure_stage=ReplayStage.EVALUATE,
            exception_type="RuntimeError",
        ),
        _run(),
    )

    summary = summarize_replays(
        task_set,
        [failed, *([matched] * 146)],
        mode=ReplayMode.GATE,
        seed=100,
        repetitions=3,
        workers=1,
    )

    assert summary.initial_state_match_rate == pytest.approx(146 / 147)
    assert summary.execution_failure_count == 1
    assert summary.exception_count == 1
    assert summary.gate_passed is False


def test_summary_is_immutable_and_contains_no_raw_ids_paths_or_payload_fields(
    tmp_path: Path,
) -> None:
    task_set = build_task_set({"train": ["private-task-id"]})
    result = _task_result(_run(), _run())
    summary = summarize_replays(
        task_set,
        [result],
        mode=ReplayMode.SMOKE,
        seed=100,
        repetitions=2,
        workers=1,
    )

    payload = summary.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    forbidden_keys = {
        "task_ids",
        "task_id",
        "path",
        "code",
        "output",
        "requests",
        "api_calls",
        "arguments",
    }

    assert forbidden_keys.isdisjoint(payload)
    assert "private-task-id" not in encoded
    assert summary.gate_evaluable is False
    assert summary.gate_passed is False
    assert summary.smoke_passed is True
    with pytest.raises(FrozenInstanceError):
        summary.gate_passed = True  # type: ignore[misc]

    output = tmp_path / "reports" / "gate0a.json"
    write_summary_json(output, summary)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert exit_code_for_summary(summary) == 0


def test_nonstandard_smoke_protocol_cannot_report_lifecycle_passed() -> None:
    task_set = build_task_set({"train": ["first", "second"]})
    result = _task_result(_run(), _run())

    summary = summarize_replays(
        task_set,
        [result, result],
        mode=ReplayMode.SMOKE,
        seed=100,
        repetitions=2,
        workers=1,
    )

    assert summary.smoke_passed is False
    assert exit_code_for_summary(summary) != 0


def test_pinned_revision_check_is_exact_and_payload_free() -> None:
    require_pinned_appworld_revision(PINNED_APPWORLD_COMMIT)

    with pytest.raises(RuntimeError, match="pinned AppWorld revision") as error:
        require_pinned_appworld_revision("0" * 40)

    assert "0" * 40 not in str(error.value)
    assert PINNED_APPWORLD_COMMIT not in str(error.value)


def test_failed_formal_summary_has_nonzero_exit_code() -> None:
    task_set = _official_task_set()
    failed = _task_result(_run(success=False), _run(), _run())
    summary = summarize_replays(
        task_set,
        [failed, *([_task_result(_run(), _run(), _run())] * 146)],
        mode=ReplayMode.GATE,
        seed=100,
        repetitions=3,
        workers=1,
    )

    assert summary.gate_passed is False
    assert exit_code_for_summary(summary) != 0


def test_cli_smoke_exit_zero_writes_non_gate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_set = build_task_set({"train": ["train-task-unit"]})
    smoke_summary = summarize_replays(
        task_set,
        [_task_result(_run(), _run())],
        mode=ReplayMode.SMOKE,
        seed=100,
        repetitions=2,
        workers=1,
    )
    monkeypatch.setattr(
        replay_module,
        "current_checkout_revision",
        lambda: PINNED_APPWORLD_COMMIT,
    )
    monkeypatch.setattr(
        replay_module,
        "load_appworld_task_set",
        lambda *, smoke: task_set,
    )
    monkeypatch.setattr(
        replay_module,
        "run_replay_verification",
        lambda *args, **kwargs: smoke_summary,
    )
    output = tmp_path / "smoke.json"

    exit_code = replay_module.main(["--smoke", "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["mode"] == "smoke"
    assert payload["gate_evaluable"] is False
    assert payload["gate_passed"] is False


def test_cli_script_is_a_thin_delegating_entry_point() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "scripts/verify_appworld_replay.py"

    assert script.is_file()
    content = script.read_text(encoding="utf-8")
    assert "from toolshift.benchmarks.appworld_replay import main" in content
    assert "raise SystemExit(main())" in content
