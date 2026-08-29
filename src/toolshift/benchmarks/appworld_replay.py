"""Deterministic, privacy-preserving AppWorld oracle replay verification.

The module deliberately has no import-time dependency on AppWorld. The real
benchmark package is imported only when a caller creates an AppWorld context.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from toolshift.types import manifest_sha256

PINNED_APPWORLD_COMMIT = "a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
FORMAL_SEED = 100
FORMAL_REPETITIONS = 3
MIN_GATE_TASKS = 100
OFFICIAL_SPLIT_COUNTS = {"dev": 57, "train": 90}
OFFICIAL_TASK_COUNT = sum(OFFICIAL_SPLIT_COUNTS.values())
_ALLOWED_SPLITS = frozenset(OFFICIAL_SPLIT_COUNTS)
_EXECUTION_FAILURE_PREFIX = "Execution failed. Traceback:"
_EXCEPTION_TYPE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ReplayMode(str, Enum):
    """Evidence status of one replay-verification invocation."""

    GATE = "gate"
    SMOKE = "smoke"
    EXPLORATORY = "exploratory"


class ReplayStage(str, Enum):
    """Payload-free lifecycle stages used for fail-closed accounting."""

    CREATE_WORLD = "create_world"
    INITIAL_STATE = "initial_state"
    EXECUTE_ORACLE = "execute_oracle"
    TRACE_FINGERPRINT = "trace_fingerprint"
    FINAL_STATE = "final_state"
    EVALUATE = "evaluate"
    CLEANUP = "cleanup"


@dataclass(frozen=True, slots=True)
class TaskSet:
    """Validated task identifiers retained only inside the execution process."""

    task_ids_by_split: Mapping[str, Sequence[str]]
    task_ids: tuple[str, ...] = field(init=False)
    split_counts: tuple[tuple[str, int], ...] = field(init=False)
    task_set_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        normalized, task_ids, split_counts, fingerprint = _derive_task_set_fields(
            self.task_ids_by_split
        )
        object.__setattr__(self, "task_ids_by_split", normalized)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "split_counts", split_counts)
        object.__setattr__(self, "task_set_sha256", fingerprint)

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(split_name for split_name, _ in self.split_counts)


@dataclass(frozen=True, slots=True)
class ReplayRun:
    """Payload-free fingerprints and status for one fresh AppWorld instance."""

    initial_state_sha256: str | None
    final_state_sha256: str | None
    evaluator_sha256: str | None
    trace_sha256: str | None
    oracle_success: bool
    execution_failed: bool
    failure_stage: ReplayStage | None = None
    exception_type: str | None = None

    def __post_init__(self) -> None:
        self._validate_integrity()

    def _validate_integrity(self) -> None:
        for field_name in (
            "initial_state_sha256",
            "final_state_sha256",
            "evaluator_sha256",
            "trace_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str)
                or _SHA256_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(
                    f"{field_name} must be a lowercase SHA-256 fingerprint or None"
                )
        if not isinstance(self.oracle_success, bool):
            raise ValueError("oracle_success must be a boolean")
        if not isinstance(self.execution_failed, bool):
            raise ValueError("execution_failed must be a boolean")
        if self.failure_stage is not None and not isinstance(
            self.failure_stage, ReplayStage
        ):
            raise ValueError("failure_stage must be a ReplayStage or None")
        if self.exception_type is not None and _EXCEPTION_TYPE_PATTERN.fullmatch(
            self.exception_type
        ) is None:
            raise ValueError("exception_type must be a payload-free class name")
        if self.execution_failed and self.oracle_success:
            raise ValueError("failed replay runs cannot set oracle_success=True")
        if self.execution_failed and self.failure_stage is None:
            raise ValueError("failed replay runs must identify a lifecycle stage")
        if not self.execution_failed and (
            self.failure_stage is not None or self.exception_type is not None
        ):
            raise ValueError("successful replay runs cannot contain failure metadata")
        if not self.execution_failed and any(
            getattr(self, field_name) is None
            for field_name in (
                "initial_state_sha256",
                "final_state_sha256",
                "evaluator_sha256",
                "trace_sha256",
            )
        ):
            raise ValueError("completed replay runs require all four fingerprints")


@dataclass(frozen=True, slots=True)
class TaskReplayResult:
    """Anonymous repeated-oracle result for one task."""

    runs: tuple[ReplayRun, ...]

    def __post_init__(self) -> None:
        try:
            runs = tuple(self.runs)
        except TypeError as error:
            raise ValueError("runs must be an iterable of ReplayRun values") from error
        if len(runs) < 2:
            raise ValueError("a task replay requires at least two fresh runs")
        if any(not isinstance(run, ReplayRun) for run in runs):
            raise ValueError("runs must contain only ReplayRun values")
        for run in runs:
            run._validate_integrity()
        object.__setattr__(self, "runs", runs)

    def _all_match(self, field_name: str) -> bool:
        if any(run.execution_failed for run in self.runs):
            return False
        values = tuple(getattr(run, field_name) for run in self.runs)
        return all(value is not None for value in values) and len(set(values)) == 1

    @property
    def initial_state_matches(self) -> bool:
        return self._all_match("initial_state_sha256")

    @property
    def final_state_matches(self) -> bool:
        return self._all_match("final_state_sha256")

    @property
    def evaluator_matches(self) -> bool:
        return self._all_match("evaluator_sha256")

    @property
    def trace_matches(self) -> bool:
        return self._all_match("trace_sha256")

    @property
    def all_oracles_succeeded(self) -> bool:
        return all(
            run.oracle_success and not run.execution_failed for run in self.runs
        )

    @property
    def is_consistent(self) -> bool:
        """Return true only when every required channel agrees without failures."""

        return (
            self.initial_state_matches
            and self.final_state_matches
            and self.evaluator_matches
            and self.trace_matches
            and self.all_oracles_succeeded
            and self.execution_failure_count == 0
            and self.exception_count == 0
        )

    @property
    def execution_failure_count(self) -> int:
        return sum(run.execution_failed for run in self.runs)

    @property
    def exception_count(self) -> int:
        return sum(run.exception_type is not None for run in self.runs)


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Aggregate-only report safe to persist outside the protected benchmark."""

    mode: ReplayMode
    pinned_appworld_commit: str
    splits: tuple[str, ...]
    split_counts: tuple[tuple[str, int], ...]
    task_set_sha256: str
    task_count: int
    repetitions: int
    episode_count: int
    seed: int
    workers: int
    initial_state_match_rate: float
    final_state_match_rate: float
    evaluator_match_rate: float
    trace_match_rate: float
    task_consistency_rate: float
    oracle_success_rate: float
    execution_failure_count: int
    exception_count: int
    failure_counts: tuple[tuple[str, int], ...]
    gate_evaluable: bool
    gate_passed: bool
    smoke_passed: bool

    def to_dict(self) -> dict[str, object]:
        """Return the fixed aggregate report schema without raw task payloads."""

        return {
            "schema_version": 1,
            "mode": self.mode.value,
            "pinned_appworld_commit": self.pinned_appworld_commit,
            "splits": list(self.splits),
            "split_counts": dict(self.split_counts),
            "task_set_sha256": self.task_set_sha256,
            "task_count": self.task_count,
            "repetitions": self.repetitions,
            "episode_count": self.episode_count,
            "seed": self.seed,
            "workers": self.workers,
            "initial_state_match_rate": self.initial_state_match_rate,
            "final_state_match_rate": self.final_state_match_rate,
            "evaluator_match_rate": self.evaluator_match_rate,
            "trace_match_rate": self.trace_match_rate,
            "task_consistency_rate": self.task_consistency_rate,
            "oracle_success_rate": self.oracle_success_rate,
            "execution_failure_count": self.execution_failure_count,
            "exception_count": self.exception_count,
            "failure_counts": dict(self.failure_counts),
            "gate_evaluable": self.gate_evaluable,
            "gate_passed": self.gate_passed,
            "smoke_passed": self.smoke_passed,
        }


WorldContextFactory = Callable[..., AbstractContextManager[Any]]


def _validate_task_id(task_id: object) -> str:
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task IDs must be non-blank strings")
    return task_id


def _derive_task_set_fields(
    task_ids_by_split: Mapping[str, Sequence[str]],
) -> tuple[
    Mapping[str, Sequence[str]],
    tuple[str, ...],
    tuple[tuple[str, int], ...],
    str,
]:
    if not isinstance(task_ids_by_split, Mapping) or not task_ids_by_split:
        raise ValueError("task split mapping must be non-empty")
    split_names = tuple(sorted(task_ids_by_split))
    invalid_splits = set(split_names) - _ALLOWED_SPLITS
    if invalid_splits:
        raise ValueError("only train/dev splits are allowed for replay verification")

    normalized_by_split: dict[str, tuple[str, ...]] = {}
    all_task_ids: list[str] = []
    for split_name in split_names:
        raw_ids = task_ids_by_split[split_name]
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            raise ValueError("task IDs for each split must be a sequence")
        task_ids = tuple(_validate_task_id(task_id) for task_id in raw_ids)
        if not task_ids:
            raise ValueError("task IDs for each split must be non-empty")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task IDs must be unique within each split")
        normalized_by_split[split_name] = tuple(sorted(task_ids))
        all_task_ids.extend(task_ids)

    if len(all_task_ids) != len(set(all_task_ids)):
        raise ValueError("task IDs must be unique across train/dev splits")
    sorted_task_ids = tuple(sorted(all_task_ids))
    fingerprint_payload = {
        split_name: list(normalized_by_split[split_name])
        for split_name in split_names
    }
    split_counts = tuple(
        (split_name, len(normalized_by_split[split_name]))
        for split_name in split_names
    )
    return (
        MappingProxyType(dict(normalized_by_split)),
        sorted_task_ids,
        split_counts,
        manifest_sha256(fingerprint_payload),
    )


def _require_task_set_integrity(task_set: object) -> TaskSet:
    if not isinstance(task_set, TaskSet):
        raise ValueError("task_set must be a validated TaskSet")
    try:
        normalized, task_ids, split_counts, fingerprint = _derive_task_set_fields(
            task_set.task_ids_by_split
        )
    except (TypeError, ValueError) as error:
        raise ValueError("task set integrity validation failed") from error
    if (
        dict(task_set.task_ids_by_split) != dict(normalized)
        or task_set.task_ids != task_ids
        or task_set.split_counts != split_counts
        or task_set.task_set_sha256 != fingerprint
    ):
        raise ValueError("task set integrity validation failed")
    return task_set


def build_task_set(task_ids_by_split: Mapping[str, Sequence[str]]) -> TaskSet:
    """Validate train/dev IDs and produce an order-stable private task set."""

    return TaskSet(task_ids_by_split)


def canonical_state_sha256(models: Any) -> str:
    """Hash AppWorld state using fresh per-record hashes, including empty tables."""

    clear_hash_cache = getattr(models, "clear_ids_record_hashes", None)
    ids_record_hashes = getattr(models, "ids_record_hashes", None)
    if not callable(clear_hash_cache) or not callable(ids_record_hashes):
        raise ValueError("models must provide AppWorld record-hash interfaces")
    clear_hash_cache()

    try:
        app_names = tuple(sorted(models.keys()))
    except (AttributeError, TypeError) as error:
        raise ValueError("models must expose sortable app names") from error

    projection: list[dict[str, object]] = []
    for app_name in app_names:
        if not isinstance(app_name, str):
            raise ValueError("AppWorld app names must be strings")
        app_models = models[app_name]
        sql_model = getattr(app_models, "SQLModel", None)
        model_names_function = getattr(sql_model, "model_names", None)
        if not callable(model_names_function):
            raise ValueError("AppWorld app models must expose SQLModel.model_names")
        try:
            model_names = tuple(sorted(model_names_function()))
        except TypeError as error:
            raise ValueError("AppWorld model names must be sortable") from error

        model_projection: list[dict[str, object]] = []
        for model_name in model_names:
            if not isinstance(model_name, str):
                raise ValueError("AppWorld model names must be strings")
            if model_name.endswith("ModelHash"):
                continue
            raw_rows = ids_record_hashes(app_name, model_name)
            normalized_rows: list[tuple[object, object]] = []
            for row in raw_rows:
                if not isinstance(row, (list, tuple)) or len(row) != 2:
                    raise ValueError("record hashes must contain (id, hash) pairs")
                record_id, record_hash = row
                if isinstance(record_id, bool) or not isinstance(record_id, int):
                    raise ValueError("record IDs must be integers")
                if not isinstance(record_hash, str) or not record_hash.strip():
                    raise ValueError("record hashes must be non-empty strings")
                normalized_rows.append((record_id, record_hash))
            rows = sorted(normalized_rows)
            model_projection.append({"model": model_name, "records": rows})
        projection.append({"app": app_name, "models": model_projection})
    return manifest_sha256(projection)


def evaluator_sha256(tracker: Any) -> str:
    """Fingerprint evaluator scores without requirements, labels, or traces."""

    to_dict = getattr(tracker, "to_dict", None)
    if not callable(to_dict):
        raise ValueError("tracker must provide to_dict")
    stats = to_dict(stats_only=True)
    if not isinstance(stats, Mapping):
        raise ValueError("tracker stats must be a mapping")
    failures = getattr(tracker, "failures", ())
    if not isinstance(failures, (list, tuple)):
        raise ValueError("tracker failures must be a sequence")
    no_op_pass_failure_count = sum(
        isinstance(failure, Mapping) and failure.get("label") == "no_op_pass"
        for failure in failures
    )
    score = {
        "pass_count": tracker.pass_count,
        "fail_count": tracker.fail_count,
        "total_count": tracker.total_count,
        "num_tests": tracker.num_tests,
        "pass_percentage": tracker.pass_percentage,
        "success": tracker.success,
        "difficulty": tracker.difficulty,
        "no_op_pass_failure_count": no_op_pass_failure_count,
    }
    return manifest_sha256({"stats": dict(stats), "score": score})


def _trace_sha256(requests: Any) -> str:
    if not isinstance(requests, list) or not requests:
        raise ValueError("AppWorld request trace must be a non-empty list")
    for request in requests:
        if not isinstance(request, Mapping):
            raise ValueError("AppWorld request trace items must be mappings")
        method = request.get("method")
        url = request.get("url")
        data = request.get("data")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("AppWorld request method must be non-blank text")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("AppWorld request URL must be non-blank text")
        if not isinstance(data, Mapping):
            raise ValueError("AppWorld request data must be a mapping")
    return manifest_sha256(requests)


@contextmanager
def managed_appworld_context(appworld_class: type[Any], **kwargs: object) -> Any:
    """Manage one instance with class-level cleanup only as a failure fallback."""

    close_all = getattr(appworld_class, "close_all", None)
    if not callable(close_all):
        raise ValueError("AppWorld class must provide close_all")
    try:
        world = appworld_class(**kwargs)
    except BaseException:
        try:
            close_all()
        except BaseException as cleanup_error:
            raise RuntimeError(
                "AppWorld global cleanup failed after constructor failure"
            ) from cleanup_error
        raise

    try:
        yield world
    finally:
        try:
            world.close()
        except BaseException:
            try:
                close_all()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "AppWorld global cleanup failed after instance close failure"
                ) from cleanup_error
            raise


@contextmanager
def appworld_world_context(**kwargs: object) -> Any:
    """Lazily import AppWorld and delegate to its guarded lifecycle manager."""

    try:
        from appworld import AppWorld
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("AppWorld runtime dependency is unavailable") from error

    with managed_appworld_context(AppWorld, **kwargs) as world:
        yield world


def _exception_type(error: Exception) -> str:
    exception_type = type(error).__name__
    if _EXCEPTION_TYPE_PATTERN.fullmatch(exception_type) is None:
        return "Exception"
    return exception_type


def _execute_single_replay(
    task_id: str,
    *,
    seed: int,
    experiment_name: str,
    world_context_factory: WorldContextFactory,
) -> ReplayRun:
    stage = ReplayStage.CREATE_WORLD
    initial_digest: str | None = None
    final_digest: str | None = None
    evaluation_digest: str | None = None
    trace_digest: str | None = None
    oracle_success = False
    execution_failed = False
    failure_stage: ReplayStage | None = None

    try:
        with world_context_factory(
            task_id=task_id,
            experiment_name=experiment_name,
            ground_truth_mode="full",
            raise_on_failure=False,
            random_seed=seed,
        ) as world:
            stage = ReplayStage.INITIAL_STATE
            initial_digest = canonical_state_sha256(world.models)

            stage = ReplayStage.EXECUTE_ORACLE
            ground_truth = world.task.ground_truth
            compiled_solution_code = ground_truth.compiled_solution_code
            if not isinstance(compiled_solution_code, str):
                raise ValueError("compiled solution must be text")
            execution_output = world.execute(
                compiled_solution_code + "\nsolution(apis, requester)"
            )
            if not isinstance(execution_output, str):
                raise ValueError("AppWorld execute output must be text")
            if execution_output.startswith(_EXECUTION_FAILURE_PREFIX):
                execution_failed = True
                failure_stage = ReplayStage.EXECUTE_ORACLE
            else:
                stage = ReplayStage.TRACE_FINGERPRINT
                trace_digest = _trace_sha256(world.requester.requests)

                stage = ReplayStage.FINAL_STATE
                final_digest = canonical_state_sha256(world.models)

                stage = ReplayStage.EVALUATE
                tracker = world.evaluate(suppress_errors=False)
                evaluation_digest = evaluator_sha256(tracker)
                oracle_success = bool(tracker.success)

            stage = ReplayStage.CLEANUP
    except Exception as error:
        return ReplayRun(
            initial_state_sha256=initial_digest,
            final_state_sha256=final_digest,
            evaluator_sha256=evaluation_digest,
            trace_sha256=trace_digest,
            oracle_success=False,
            execution_failed=True,
            failure_stage=stage,
            exception_type=_exception_type(error),
        )

    return ReplayRun(
        initial_state_sha256=initial_digest,
        final_state_sha256=final_digest,
        evaluator_sha256=evaluation_digest,
        trace_sha256=trace_digest,
        oracle_success=oracle_success,
        execution_failed=execution_failed,
        failure_stage=failure_stage,
        exception_type=None,
    )


def run_task_repetitions(
    task_id: str,
    *,
    seed: int,
    repetitions: int,
    world_context_factory: WorldContextFactory = appworld_world_context,
) -> TaskReplayResult:
    """Replay one oracle in independent fresh worlds without returning its ID."""

    _validate_task_id(task_id)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("repetitions must be an integer of at least two")
    runs = tuple(
        _execute_single_replay(
            task_id,
            seed=seed,
            experiment_name=f"toolshift-gate0a-{uuid.uuid4().hex}",
            world_context_factory=world_context_factory,
        )
        for _ in range(repetitions)
    )
    return TaskReplayResult(runs=runs)


def _rate(values: Sequence[bool]) -> float:
    if not values:
        raise ValueError("cannot compute a rate over an empty sequence")
    return sum(values) / len(values)


def _failure_counts(results: Sequence[TaskReplayResult]) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    for result in results:
        for run in result.runs:
            if not run.execution_failed or run.failure_stage is None:
                continue
            failure_kind = run.exception_type or "execution_traceback"
            counts[f"{run.failure_stage.value}:{failure_kind}"] += 1
    return tuple(sorted(counts.items()))


def _is_official_gate_protocol(
    task_set: TaskSet,
    *,
    mode: ReplayMode,
    seed: int,
    repetitions: int,
    workers: int,
) -> bool:
    return (
        mode is ReplayMode.GATE
        and len(task_set.task_ids) >= MIN_GATE_TASKS
        and len(task_set.task_ids) == OFFICIAL_TASK_COUNT
        and dict(task_set.split_counts) == OFFICIAL_SPLIT_COUNTS
        and seed == FORMAL_SEED
        and repetitions >= FORMAL_REPETITIONS
        and workers == 1
    )


def _require_replay_results_integrity(
    results: object,
    *,
    repetitions: int,
) -> tuple[TaskReplayResult, ...]:
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise ValueError("results must be a sequence of TaskReplayResult values")
    validated: list[TaskReplayResult] = []
    for result in results:
        if not isinstance(result, TaskReplayResult):
            raise ValueError("results must contain only TaskReplayResult values")
        if not isinstance(result.runs, tuple) or len(result.runs) != repetitions:
            raise ValueError("replay result integrity validation failed")
        try:
            TaskReplayResult(result.runs)
        except (TypeError, ValueError) as error:
            raise ValueError("replay result integrity validation failed") from error
        validated.append(result)
    return tuple(validated)


def summarize_replays(
    task_set: TaskSet,
    results: Sequence[TaskReplayResult],
    *,
    mode: ReplayMode,
    seed: int,
    repetitions: int,
    workers: int,
) -> ReplaySummary:
    """Aggregate anonymous task results under the preregistered Gate 0a rule."""

    task_set = _require_task_set_integrity(task_set)
    if not isinstance(mode, ReplayMode):
        raise ValueError("mode must be a ReplayMode")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 2
    ):
        raise ValueError("repetitions must be an integer of at least two")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    results = _require_replay_results_integrity(results, repetitions=repetitions)
    if len(results) != len(task_set.task_ids):
        raise ValueError("one anonymous result is required for every task")

    task_count = len(results)
    episode_count = task_count * repetitions
    initial_match_rate = _rate([result.initial_state_matches for result in results])
    final_match_rate = _rate([result.final_state_matches for result in results])
    evaluator_match_rate = _rate([result.evaluator_matches for result in results])
    trace_match_rate = _rate([result.trace_matches for result in results])
    task_consistency_rate = _rate([result.is_consistent for result in results])
    execution_failure_count = sum(
        result.execution_failure_count for result in results
    )
    exception_count = sum(result.exception_count for result in results)
    oracle_success_rate = sum(
        run.oracle_success and not run.execution_failed
        for result in results
        for run in result.runs
    ) / episode_count
    gate_evaluable = _is_official_gate_protocol(
        task_set,
        mode=mode,
        seed=seed,
        repetitions=repetitions,
        workers=workers,
    )
    metrics_pass = (
        initial_match_rate == 1.0
        and final_match_rate >= 0.99
        and evaluator_match_rate >= 0.99
        and trace_match_rate >= 0.99
        and task_consistency_rate >= 0.99
        and oracle_success_rate == 1.0
        and execution_failure_count == 0
        and exception_count == 0
    )
    smoke_protocol = (
        mode is ReplayMode.SMOKE
        and dict(task_set.split_counts) == {"train": 1}
        and seed == FORMAL_SEED
        and repetitions == 2
        and workers == 1
    )
    smoke_passed = smoke_protocol and metrics_pass
    return ReplaySummary(
        mode=mode,
        pinned_appworld_commit=PINNED_APPWORLD_COMMIT,
        splits=task_set.split_names,
        split_counts=task_set.split_counts,
        task_set_sha256=task_set.task_set_sha256,
        task_count=task_count,
        repetitions=repetitions,
        episode_count=episode_count,
        seed=seed,
        workers=workers,
        initial_state_match_rate=initial_match_rate,
        final_state_match_rate=final_match_rate,
        evaluator_match_rate=evaluator_match_rate,
        trace_match_rate=trace_match_rate,
        task_consistency_rate=task_consistency_rate,
        oracle_success_rate=oracle_success_rate,
        execution_failure_count=execution_failure_count,
        exception_count=exception_count,
        failure_counts=_failure_counts(results),
        gate_evaluable=gate_evaluable,
        gate_passed=gate_evaluable and metrics_pass,
        smoke_passed=smoke_passed,
    )


def _run_default_task(arguments: tuple[str, int, int]) -> TaskReplayResult:
    task_id, seed, repetitions = arguments
    return run_task_repetitions(task_id, seed=seed, repetitions=repetitions)


def run_replay_verification(
    task_set: TaskSet,
    *,
    mode: ReplayMode,
    seed: int,
    repetitions: int,
    workers: int = 1,
    world_context_factory: WorldContextFactory = appworld_world_context,
) -> ReplaySummary:
    """Run replay groups serially, or explicitly as non-gate exploratory work."""

    task_set = _require_task_set_integrity(task_set)
    if workers == 1:
        results = [
            run_task_repetitions(
                task_id,
                seed=seed,
                repetitions=repetitions,
                world_context_factory=world_context_factory,
            )
            for task_id in task_set.task_ids
        ]
    else:
        if world_context_factory is not appworld_world_context:
            raise ValueError("parallel verification requires the default AppWorld factory")
        from concurrent.futures import ProcessPoolExecutor

        arguments = (
            (task_id, seed, repetitions) for task_id in task_set.task_ids
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_default_task, arguments))
    return summarize_replays(
        task_set,
        results,
        mode=mode,
        seed=seed,
        repetitions=repetitions,
        workers=workers,
    )


def require_pinned_appworld_revision(actual_revision: str) -> None:
    """Reject any checkout other than the preregistered AppWorld revision."""

    if actual_revision != PINNED_APPWORLD_COMMIT:
        raise RuntimeError("current checkout is not the pinned AppWorld revision")


def current_checkout_revision() -> str:
    """Read HEAD from the current working directory without exposing its path."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not verify the AppWorld checkout revision") from error
    revision = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("AppWorld checkout returned an invalid revision")
    return revision


def load_appworld_task_set(*, smoke: bool) -> TaskSet:
    """Lazily load only train/dev identifiers from the pinned AppWorld package."""

    try:
        from appworld import load_task_ids
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("AppWorld runtime dependency is unavailable") from error

    train_ids = tuple(load_task_ids("train"))
    if smoke:
        if not train_ids:
            raise RuntimeError("AppWorld train split is empty")
        return build_task_set({"train": [sorted(train_ids)[0]]})
    dev_ids = tuple(load_task_ids("dev"))
    return build_task_set({"train": train_ids, "dev": dev_ids})


def write_summary_json(path: str | Path, summary: ReplaySummary) -> None:
    """Atomically write one aggregate-only JSON report with private permissions."""

    if not isinstance(summary, ReplaySummary):
        raise ValueError("summary must be a ReplaySummary")
    output_path = Path(path)
    if not output_path.name:
        raise ValueError("output path must name a JSON file")
    if output_path.is_symlink():
        raise ValueError("output path must not be a symbolic link")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            os.fchmod(temporary.fileno(), 0o600)
            json.dump(
                summary.to_dict(),
                temporary,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def exit_code_for_summary(summary: ReplaySummary) -> int:
    """Return success for a passed gate or lifecycle smoke, never for a failed gate."""

    if summary.mode is ReplayMode.SMOKE:
        return 0 if summary.smoke_passed else 1
    if summary.mode is ReplayMode.GATE:
        return 0 if summary.gate_passed else 1
    return 0 if summary.execution_failure_count == 0 else 1


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify deterministic AppWorld oracle replay without persisting payloads."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="aggregate JSON report path outside the protected checkout",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one train task twice; this can never pass Gate 0a",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes; values above one are exploratory, not Gate evidence",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=FORMAL_REPETITIONS,
        help="fresh worlds per task in formal mode (minimum: 3)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for formal Gate 0a and explicit lifecycle smoke runs."""

    arguments = _argument_parser().parse_args(argv)
    if arguments.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not arguments.smoke and arguments.repetitions < FORMAL_REPETITIONS:
        raise SystemExit("formal replay requires at least 3 repetitions")

    require_pinned_appworld_revision(current_checkout_revision())
    task_set = load_appworld_task_set(smoke=arguments.smoke)
    if arguments.smoke:
        mode = ReplayMode.SMOKE
        repetitions = 2
        workers = 1
    else:
        mode = ReplayMode.GATE if arguments.workers == 1 else ReplayMode.EXPLORATORY
        repetitions = arguments.repetitions
        workers = arguments.workers
    summary = run_replay_verification(
        task_set,
        mode=mode,
        seed=FORMAL_SEED,
        repetitions=repetitions,
        workers=workers,
    )
    write_summary_json(arguments.output, summary)
    return exit_code_for_summary(summary)


__all__ = [
    "FORMAL_REPETITIONS",
    "FORMAL_SEED",
    "MIN_GATE_TASKS",
    "OFFICIAL_TASK_COUNT",
    "PINNED_APPWORLD_COMMIT",
    "ReplayMode",
    "ReplayRun",
    "ReplayStage",
    "ReplaySummary",
    "TaskReplayResult",
    "TaskSet",
    "appworld_world_context",
    "build_task_set",
    "canonical_state_sha256",
    "current_checkout_revision",
    "evaluator_sha256",
    "exit_code_for_summary",
    "load_appworld_task_set",
    "main",
    "managed_appworld_context",
    "require_pinned_appworld_revision",
    "run_replay_verification",
    "run_task_repetitions",
    "summarize_replays",
    "write_summary_json",
]
