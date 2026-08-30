"""Synthetic tests for paired AppWorld Gate 0 contract admission."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from toolshift.adapters import build_appworld_adapter
from toolshift.benchmarks.appworld_runtime import _CapturedOraclePlan
from toolshift.contracts import StateCase, StateEvidence, TraceCase, TraceEvidence
from toolshift.transforms import (
    ParameterGroupRule,
    apply_parameter_restructure,
    apply_rename,
)
from toolshift.types import JSONValue, canonical_json_bytes


def _tool(
    name: str,
    properties: Mapping[str, object],
    required: list[str],
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Synthetic tool.",
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": required,
            },
        },
    }


def _catalog() -> list[dict[str, object]]:
    return [
        _tool(
            "notes__create_note",
            {
                "title": {"type": "string", "minLength": 1},
                "priority": {"type": "integer"},
            },
            ["title"],
        ),
        _tool(
            "notes__search_notes",
            {"query": {"type": "string", "minLength": 1}},
            ["query"],
        ),
    ]


class _SQLModel:
    @staticmethod
    def model_names() -> tuple[str, ...]:
        return ("SyntheticState",)


class _AppModels:
    SQLModel = _SQLModel


class _Models(Mapping[str, _AppModels]):
    def __init__(self) -> None:
        self._apps = {"notes": _AppModels()}
        self._state: list[bytes] = []

    def __getitem__(self, key: str) -> _AppModels:
        return self._apps[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._apps)

    def __len__(self) -> int:
        return len(self._apps)

    def clear_ids_record_hashes(self) -> None:
        return None

    def ids_record_hashes(
        self,
        app_name: str,
        model_name: str,
    ) -> tuple[tuple[int, str], ...]:
        assert app_name == "notes"
        assert model_name == "SyntheticState"
        digest = hashlib.sha256(b"".join(self._state)).hexdigest()
        return ((1, digest),)

    def apply(self, call: Mapping[str, object]) -> None:
        self._state.append(canonical_json_bytes(call))


class _Tracker:
    pass_count = 1
    fail_count = 0
    total_count = 1
    num_tests = 1
    pass_percentage = 100.0
    success = True
    difficulty = 1
    failures: tuple[object, ...] = ()

    @staticmethod
    def to_dict(*, stats_only: bool) -> dict[str, object]:
        assert stats_only is True
        return {"pass_count": 1, "total_count": 1}


class _Requester:
    def __init__(self, models: _Models, *, corrupt_observation: bool = False) -> None:
        self._models = models
        self._corrupt_observation = corrupt_observation

    def request(
        self,
        *,
        _app_name: str,
        _api_name: str,
        **arguments: object,
    ) -> dict[str, object]:
        call = {
            "name": f"{_app_name}__{_api_name}",
            "arguments": arguments,
        }
        self._models.apply(call)
        return {
            "ok": not self._corrupt_observation,
            "call_index": len(self._models._state) - 1,
        }


class _World:
    def __init__(self, *, corrupt_observation: bool = False) -> None:
        self.models = _Models()
        self.requester = _Requester(
            self.models,
            corrupt_observation=corrupt_observation,
        )
        self.save_calls = 0
        self.evaluate_calls = 0

    def save(self) -> None:
        self.save_calls += 1

    def evaluate(self, *, suppress_errors: bool) -> _Tracker:
        assert suppress_errors is False
        self.evaluate_calls += 1
        return _Tracker()


class _WorldFactory:
    def __init__(
        self,
        *,
        corrupt_candidate: bool = False,
        cleanup_failure_role: str | None = None,
        reuse_world: bool = False,
        reset_mismatch: bool = False,
    ) -> None:
        self.corrupt_candidate = corrupt_candidate
        self.cleanup_failure_role = cleanup_failure_role
        self.reuse_world = reuse_world
        self.reset_mismatch = reset_mismatch
        self.open_roles: list[str] = []
        self.close_roles: list[str] = []
        self.live_count = 0
        self.max_live_count = 0
        self.worlds: list[tuple[str, _World]] = []
        self._shared_world: _World | None = None

    @contextmanager
    def __call__(self, *, role: str) -> Iterator[_World]:
        if self.reuse_world and self._shared_world is not None:
            world = self._shared_world
        else:
            world = _World(corrupt_observation=self.corrupt_candidate and role == "candidate")
            if self.reuse_world:
                self._shared_world = world
        if self.reset_mismatch and role == "candidate-reset":
            world.models.apply({"name": "synthetic__reset_canary", "arguments": {"value": 1}})
        self.open_roles.append(role)
        self.live_count += 1
        self.max_live_count = max(self.max_live_count, self.live_count)
        self.worlds.append((role, world))
        try:
            yield world
        finally:
            self.live_count -= 1
            self.close_roles.append(role)
            if role == self.cleanup_failure_role:
                raise RuntimeError("PRIVATE_CLEANUP_CANARY")


class _SmokeApiDocs:
    def __init__(self, catalog: list[dict[str, object]], events: list[object]) -> None:
        self._catalog = catalog
        self._events = events

    def function_calling(self) -> list[dict[str, object]]:
        self._events.append("catalog")
        return self._catalog


class _SmokeRequester:
    def __init__(self, models: _Models, events: list[object]) -> None:
        self._models = models
        self._events = events
        self.requests: list[object] = []

    def request(
        self,
        _app_name: str,
        _api_name: str,
        client: object = None,
        raise_on_failure: object = None,
        show: object = False,
        track: object = True,
        **data: object,
    ) -> dict[str, object]:
        del client, raise_on_failure, show
        call = {"name": f"{_app_name}__{_api_name}", "arguments": data}
        self.post("/synthetic", track=track)
        self._models.apply(call)
        self.requests.append({"method": "POST"})
        return {"ok": True, "call_index": len(self._models._state) - 1}

    def get(self, *args: object, **kwargs: object) -> dict[str, bool]:
        del args, kwargs
        return {"ok": True}

    def post(self, *args: object, **kwargs: object) -> dict[str, bool]:
        del args, kwargs
        return {"ok": True}

    def put(self, *args: object, **kwargs: object) -> dict[str, bool]:
        del args, kwargs
        return {"ok": True}

    def patch(self, *args: object, **kwargs: object) -> dict[str, bool]:
        del args, kwargs
        return {"ok": True}

    def delete(self, *args: object, **kwargs: object) -> dict[str, bool]:
        del args, kwargs
        return {"ok": True}


class _SmokeWorld:
    def __init__(
        self,
        catalog: list[dict[str, object]],
        oracle_calls: tuple[Mapping[str, object], ...],
        events: list[object],
    ) -> None:
        self.models = _Models()
        self.requester = _SmokeRequester(self.models, events)
        self.task = type("Task", (), {})()
        self.task.api_docs = _SmokeApiDocs(catalog, events)
        self.task.ground_truth = type("GroundTruth", (), {})()
        self.task.ground_truth.compiled_solution_code = "def solution(apis, requester): return None"
        self._oracle_calls = oracle_calls
        self.save_calls = 0

    def execute(self, code: str) -> str:
        assert code.endswith("\nsolution(apis, requester)")
        for call in self._oracle_calls:
            name = call["name"]
            arguments = call["arguments"]
            assert isinstance(name, str)
            assert isinstance(arguments, Mapping)
            app_name, api_name = name.split("__", 1)
            self.requester.request(app_name, api_name, **arguments)
        return "ok"

    def save(self) -> None:
        self.save_calls += 1

    def evaluate(self, *, suppress_errors: bool) -> _Tracker:
        assert suppress_errors is False
        return _Tracker()


class _SmokeContextFactory:
    def __init__(
        self,
        task_specs: Mapping[
            str,
            tuple[list[dict[str, object]], tuple[Mapping[str, object], ...]],
        ],
    ) -> None:
        self.task_specs = task_specs
        self.calls: list[dict[str, object]] = []
        self.events: list[object] = []
        self.worlds: list[_SmokeWorld] = []
        self.live_count = 0
        self.max_live_count = 0

    @contextmanager
    def __call__(self, **kwargs: object) -> Iterator[_SmokeWorld]:
        self.calls.append(dict(kwargs))
        task_id = kwargs["task_id"]
        assert isinstance(task_id, str)
        catalog, oracle_calls = self.task_specs[task_id]
        world = _SmokeWorld(catalog, oracle_calls, self.events)
        self.worlds.append(world)
        self.live_count += 1
        self.max_live_count = max(self.max_live_count, self.live_count)
        try:
            yield world
        finally:
            self.live_count -= 1


def _source_and_calls():
    source = build_appworld_adapter(_catalog())
    plan = _CapturedOraclePlan(
        (
            {
                "name": "notes__create_note",
                "arguments": {"title": "alpha", "priority": 1},
            },
            {"name": "notes__search_notes", "arguments": {"query": "alpha"}},
        )
    )
    return source, plan


def _candidate(kind: str):
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, plan = _source_and_calls()
    if kind == "clean":
        return (
            source,
            source,
            plan,
            getattr(gate0_module, "_identity_canonical_call_to_surface", lambda call: call),
        )
    if kind == "rename":
        candidate = apply_rename(
            source,
            tool_name_mapping={"notes__create_note": "toolshift_alias_0000"},
            seed=100,
        )
        return source, candidate, plan, candidate.transform.canonical_call_to_surface
    if kind == "restructure":
        candidate = apply_parameter_restructure(
            source,
            rules=(
                ParameterGroupRule(
                    "notes__create_note",
                    "toolshift_group_0000",
                    ("title",),
                ),
            ),
            seed=100,
        )
        return source, candidate, plan, candidate.transform.canonical_call_to_surface
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["clean", "rename", "restructure"])
def test_build_schema_probes_covers_candidate_inventory_from_source_witnesses(
    kind: str,
) -> None:
    from toolshift.benchmarks.appworld_gate0 import build_schema_probes

    source, candidate, _, translate = _candidate(kind)

    probes = build_schema_probes(source, candidate, translate)

    assert tuple(probe.case_id for probe in probes) == ("schema-0000", "schema-0001")
    assert {probe.surface_tool_name for probe in probes} == {
        tool.name for tool in candidate.variant.tools
    }
    assert len(probes) == len(source.variant.tools) == len(candidate.variant.tools)
    for probe in probes:
        canonical = (
            candidate.transform.surface_call_to_canonical(probe.surface_call)
            if kind != "clean"
            else probe.surface_call
        )
        assert probe.expected_actions == source.surface_to_semantic(canonical)


@pytest.mark.parametrize("kind", ["clean", "rename", "restructure"])
def test_admit_variant_pair_uses_four_serial_fresh_worlds_and_hard_admission(
    kind: str,
) -> None:
    from toolshift.benchmarks.appworld_gate0 import _admit_variant_pair

    source, candidate, plan, translate = _candidate(kind)
    factory = _WorldFactory()

    record = _admit_variant_pair(
        source_adapter=source,
        candidate_adapter=candidate,
        oracle_plan=plan,
        canonical_call_to_surface=translate,
        world_context_factory=factory,
    )

    assert factory.open_roles == [
        "reference",
        "candidate",
        "reference-reset",
        "candidate-reset",
    ]
    assert factory.close_roles == factory.open_roles
    assert factory.max_live_count == 1
    assert record.admitted_suite_count == 1
    assert record.diagnostic_codes == ()
    assert dict(record.checks_run) == {
        "schema": 2,
        "denotation": 2,
        "state": 1,
        "trace": 2,
    }
    assert {field.name for field in fields(record)} == {
        "admitted_suite_count",
        "checks_run",
        "diagnostic_codes",
    }
    assert "notes" not in repr(record)
    for role, world in factory.worlds:
        if role in {"reference", "candidate"}:
            assert world.save_calls == len(plan.native_calls)
            assert world.evaluate_calls == 1
        else:
            assert world.save_calls == 0
            assert world.evaluate_calls == 0


def test_admit_variant_pair_immediately_admits_the_same_suite_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("rename")
    factory = _WorldFactory()
    evaluated: list[object] = []
    admitted: list[object] = []
    real_evaluate = gate0_module.evaluate_contract_suite
    real_require = gate0_module.require_dataset_admission

    def evaluate_spy(*args: object, **kwargs: object) -> object:
        assert factory.live_count == 0
        denotation_cases = kwargs["denotation_cases"]
        state_case = kwargs["state_cases"][0]  # type: ignore[index]
        trace_case = kwargs["trace_cases"][0]  # type: ignore[index]
        trace_evidence = kwargs["trace_provider"](trace_case)  # type: ignore[operator]
        assert state_case.episode_id == trace_case.episode_id == "episode-0000"
        assert len(trace_evidence.candidate_steps) == len(denotation_cases)
        assert all(
            cached is registered
            for cached, registered in zip(
                trace_evidence.candidate_steps,
                denotation_cases,
                strict=True,
            )
        )
        suite = real_evaluate(*args, **kwargs)
        evaluated.append(suite)
        return suite

    def admission_spy(variant: object, suite: object) -> object:
        admitted.append(suite)
        return real_require(variant, suite)

    monkeypatch.setattr(gate0_module, "evaluate_contract_suite", evaluate_spy)
    monkeypatch.setattr(gate0_module, "require_dataset_admission", admission_spy)

    gate0_module._admit_variant_pair(
        source_adapter=source,
        candidate_adapter=candidate,
        oracle_plan=plan,
        canonical_call_to_surface=translate,
        world_context_factory=factory,
    )

    assert len(evaluated) == len(admitted) == 1
    assert admitted[0] is evaluated[0]


def test_cached_evidence_lookups_require_exact_case_identity() -> None:
    from toolshift.benchmarks.appworld_gate0 import (
        _StateEvidenceLookup,
        _TraceEvidenceLookup,
    )
    from toolshift.contracts import PhysicalCallEffect
    from toolshift.types import ExecutionTrace

    state_case = StateCase("state-0000", "episode-0000")
    state_evidence = StateEvidence(
        *(hashlib.sha256(label).hexdigest() for label in (b"i", b"i", b"f", b"f", b"i", b"i")),
        "collateral-0000",
        "collateral-0000",
    )
    trace_case = TraceCase("trace-0000", "episode-0000")
    empty_trace = ExecutionTrace((), (), ())
    trace_evidence = TraceEvidence(
        empty_trace,
        empty_trace,
        (),
        {"success": True},
        {"success": True},
        tuple[PhysicalCallEffect, ...](),
        tuple[PhysicalCallEffect, ...](),
    )
    state_lookup = _StateEvidenceLookup(((state_case, state_evidence),))
    trace_lookup = _TraceEvidenceLookup(((trace_case, trace_evidence),))

    assert state_lookup(state_case) is state_evidence
    assert trace_lookup(trace_case) is trace_evidence
    with pytest.raises(ValueError, match="registered state case"):
        state_lookup(StateCase("state-0000", "episode-0000"))
    with pytest.raises(ValueError, match="registered trace case"):
        trace_lookup(TraceCase("trace-0000", "episode-0000"))


def test_candidate_observation_mismatch_fails_before_contract_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("rename")
    factory = _WorldFactory(corrupt_candidate=True)
    monkeypatch.setattr(
        gate0_module,
        "evaluate_contract_suite",
        lambda *args, **kwargs: pytest.fail("contract suite must not be materialized"),
    )

    with pytest.raises(ValueError, match="paired AppWorld evidence failed") as error:
        gate0_module._admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=factory,
        )

    assert "notes" not in str(error.value)
    assert error.value.diagnostic_codes == (  # type: ignore[attr-defined]
        "pair.base_observation_mismatch",
    )
    assert factory.close_roles == ["reference", "candidate"]


def test_pair_record_never_contains_protected_episode_values() -> None:
    from toolshift.benchmarks.appworld_gate0 import _admit_variant_pair

    source, candidate, plan, translate = _candidate("restructure")
    record = _admit_variant_pair(
        source_adapter=source,
        candidate_adapter=candidate,
        oracle_plan=plan,
        canonical_call_to_surface=translate,
        world_context_factory=_WorldFactory(),
    )

    encoded = repr(record)
    for canary in (
        "notes__create_note",
        "toolshift_group_0000",
        "alpha",
        "PRIVATE_TASK_CANARY",
        "PRIVATE_OBSERVATION_CANARY",
    ):
        assert canary not in encoded


def test_build_schema_probes_never_asks_candidate_for_expected_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toolshift.benchmarks.appworld_gate0 import build_schema_probes

    source, candidate, _, translate = _candidate("rename")

    def forbidden_candidate_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("candidate must not manufacture expected actions")

    monkeypatch.setattr(
        type(candidate),
        "surface_to_semantic",
        forbidden_candidate_parse,
    )

    probes = build_schema_probes(source, candidate, translate)

    assert len(probes) == 2


@pytest.mark.parametrize(
    "bad_translator",
    [
        pytest.param(lambda call: call, id="lambda"),
        pytest.param(lambda call: dict(call), id="detached-wrapper"),
    ],
)
def test_schema_probe_builder_rejects_untrusted_translator_identity(
    bad_translator: Any,
) -> None:
    from toolshift.benchmarks.appworld_gate0 import build_schema_probes

    source, candidate, _, _ = _candidate("rename")

    with pytest.raises(ValueError, match="trusted canonical translator") as error:
        build_schema_probes(source, candidate, bad_translator)

    assert error.value.__cause__ is None


@pytest.mark.parametrize("failure", ["duplicate", "missing", "unknown", "nondeterministic"])
def test_schema_probe_builder_rejects_bad_trusted_translation_inventory(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from toolshift.benchmarks.appworld_gate0 import build_schema_probes

    source, candidate, plan, _ = _candidate("rename")
    transform_type = type(candidate.transform)
    original = candidate.transform.canonical_call_to_surface
    calls = 0

    def bad_translation(self: object, call: Mapping[str, JSONValue]):
        nonlocal calls
        assert self is candidate.transform
        calls += 1
        if failure == "duplicate":
            return original(plan.native_calls[0])
        if failure == "missing" and call["name"] == "notes__search_notes":
            raise RuntimeError("PRIVATE_MISSING_TRANSLATION_CANARY")
        translated = original(call)
        if failure == "unknown":
            return {"name": "private__unknown_tool", "arguments": translated["arguments"]}
        if failure == "nondeterministic" and calls % 2 == 0:
            return {"name": translated["name"], "arguments": {"private": calls}}
        return translated

    monkeypatch.setattr(transform_type, "canonical_call_to_surface", bad_translation)
    translate = candidate.transform.canonical_call_to_surface

    with pytest.raises(ValueError, match="schema probe construction failed") as error:
        build_schema_probes(source, candidate, translate)

    assert error.value.__cause__ is None
    assert "PRIVATE_" not in str(error.value)


def test_admit_variant_pair_requires_exact_captured_oracle_plan() -> None:
    from toolshift.benchmarks.appworld_gate0 import _admit_variant_pair

    source, candidate, plan, translate = _candidate("rename")

    with pytest.raises(ValueError, match="captured oracle plan") as error:
        _admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan.native_calls,
            canonical_call_to_surface=translate,
            world_context_factory=_WorldFactory(),
        )

    assert error.value.__cause__ is None


def _install_candidate_record_corruption(
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    real_execute = gate0_module.AppWorldEpisodeExecutor.execute_plan
    executions = 0

    def corrupted_execute(executor: object, calls: object) -> object:
        nonlocal executions
        record = real_execute(executor, calls)
        executions += 1
        if executions != 2:
            return record
        first = record.steps[0]
        if channel == "action":
            from toolshift.types import SemanticAction

            assert first.action.arguments == {"title": "alpha", "priority": 1}
            assert dict(first.action.arguments) == {"title": "alpha", "priority": True}
            assert first.action != SemanticAction(
                first.action.name,
                {"title": "alpha", "priority": True},
            )
            changed = replace(
                first,
                action=SemanticAction(
                    first.action.name,
                    {"title": "alpha", "priority": True},
                ),
            )
        elif channel == "base_call":
            changed = replace(
                first,
                base_call={
                    "name": first.base_call["name"],
                    "arguments": {"title": "alpha", "priority": True},
                },
            )
        elif channel == "base_observation":
            changed = replace(first, base_observation={"PRIVATE_BASE_OBSERVATION": True})
        elif channel == "surface_observation":
            changed = replace(
                first,
                surface_observation={"PRIVATE_SURFACE_OBSERVATION": True},
            )
        elif channel == "effect":
            from toolshift.contracts import PhysicalCallEffect

            first_effect = record.effects[0]
            return replace(
                record,
                effects=(
                    PhysicalCallEffect(
                        first_effect.call_index,
                        first_effect.base_call_fingerprint,
                        "mismatched-effect-0000",
                    ),
                    *record.effects[1:],
                ),
            )
        elif channel == "final":
            return replace(
                record,
                final_state_sha256=hashlib.sha256(b"mismatched-final").hexdigest(),
            )
        elif channel == "collateral":
            return replace(
                record,
                transition_digest=hashlib.sha256(b"mismatched-collateral").hexdigest(),
            )
        elif channel == "score":
            return replace(
                record,
                evaluator_score={
                    "pass_count": 2,
                    "fail_count": 0,
                    "total_count": 2,
                    "num_tests": 2,
                    "success": True,
                },
            )
        else:  # pragma: no cover - test helper guard.
            raise AssertionError(channel)
        return replace(record, steps=(changed, *record.steps[1:]))

    monkeypatch.setattr(
        gate0_module.AppWorldEpisodeExecutor,
        "execute_plan",
        corrupted_execute,
    )


@pytest.mark.parametrize(
    ("channel", "code"),
    [
        ("action", "pair.action_mismatch"),
        ("base_call", "pair.base_call_mismatch"),
        ("base_observation", "pair.base_observation_mismatch"),
        ("surface_observation", "pair.surface_observation_mismatch"),
    ],
)
def test_direct_candidate_channel_mismatch_fails_before_contracts(
    channel: str,
    code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("rename")
    _install_candidate_record_corruption(monkeypatch, channel)
    monkeypatch.setattr(
        gate0_module,
        "evaluate_contract_suite",
        lambda *args, **kwargs: pytest.fail("contracts must not run"),
    )

    with pytest.raises(ValueError, match="paired AppWorld evidence failed") as error:
        gate0_module._admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=_WorldFactory(),
        )

    assert error.value.diagnostic_codes == (code,)  # type: ignore[attr-defined]
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    ("channel", "code"),
    [
        ("effect", "trace.effect_mismatch"),
        ("final", "state.final_mismatch"),
        ("collateral", "state.collateral_mismatch"),
        ("score", "trace.score_mismatch"),
    ],
)
def test_contract_negative_never_returns_a_success_record(
    channel: str,
    code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("rename")
    _install_candidate_record_corruption(monkeypatch, channel)

    with pytest.raises(ValueError, match="paired AppWorld evidence failed") as error:
        gate0_module._admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=_WorldFactory(),
        )

    assert code in error.value.diagnostic_codes  # type: ignore[attr-defined]
    assert error.value.__cause__ is None


def test_reset_negative_never_returns_a_success_record() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("rename")

    with pytest.raises(ValueError, match="paired AppWorld evidence failed") as error:
        gate0_module._admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=_WorldFactory(reset_mismatch=True),
        )

    assert "state.candidate_reset_mismatch" in error.value.diagnostic_codes  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "role",
    ["reference", "candidate", "reference-reset", "candidate-reset"],
)
def test_cleanup_failures_are_static_and_never_return_success(role: str) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("rename")

    with pytest.raises(ValueError, match="paired AppWorld execution failed") as error:
        gate0_module._admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=_WorldFactory(cleanup_failure_role=role),
        )

    assert "PRIVATE_CLEANUP_CANARY" not in str(error.value)
    assert error.value.__cause__ is None


def test_reusing_a_world_across_roles_fails_closed() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("rename")

    with pytest.raises(ValueError, match="fresh world identity") as error:
        gate0_module._admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=_WorldFactory(reuse_world=True),
        )

    assert error.value.__cause__ is None


class _PrivateBaseException(BaseException):
    pass


class _BaseExceptionFactory(_WorldFactory):
    @contextmanager
    def __call__(self, *, role: str) -> Iterator[_World]:
        with super().__call__(role=role) as world:
            yield world
        if role == "candidate":
            raise _PrivateBaseException("PRIVATE_BASE_EXCEPTION_CANARY")


def test_baseexception_is_sanitized_after_world_cleanup() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("rename")
    factory = _BaseExceptionFactory()

    with pytest.raises(ValueError, match="paired AppWorld execution failed") as error:
        gate0_module._admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=factory,
        )

    assert "PRIVATE_BASE_EXCEPTION_CANARY" not in str(error.value)
    assert error.value.__cause__ is None
    assert factory.close_roles == ["reference", "candidate"]


def test_private_admission_error_repr_and_codes_are_payload_safe() -> None:
    from toolshift.benchmarks.appworld_gate0 import _VariantAdmissionError

    error = _VariantAdmissionError(
        "PRIVATE_ERROR_MESSAGE_CANARY",
        ("pair.action_mismatch", "PRIVATE_TASK_ID_CANARY"),
    )

    assert repr(error) == "_VariantAdmissionError(<private>)"
    assert error.diagnostic_codes == (
        "pair.action_mismatch",
        "pair.unsafe_diagnostic",
    )
    assert str(error) == "paired AppWorld execution failed"
    assert "PRIVATE" not in repr(error)
    assert "PRIVATE" not in str(error)


@pytest.mark.parametrize(
    ("unsuccessful_execution", "diagnostic_code"),
    [
        (1, "pair.reference_oracle_unsuccessful"),
        (2, "pair.candidate_oracle_unsuccessful"),
    ],
)
def test_admission_rejects_identical_well_formed_unsuccessful_scores(
    monkeypatch: pytest.MonkeyPatch,
    unsuccessful_execution: int,
    diagnostic_code: str,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source, candidate, plan, translate = _candidate("clean")
    real_execute = gate0_module.AppWorldEpisodeExecutor.execute_plan

    execution_count = 0

    def unsuccessful_execute(executor: object, calls: object) -> object:
        nonlocal execution_count
        execution_count += 1
        record = real_execute(executor, calls)
        if execution_count != unsuccessful_execution:
            return record
        return replace(
            record,
            evaluator_score={
                "pass_count": 0,
                "fail_count": 1,
                "total_count": 1,
                "num_tests": 1,
                "success": False,
            },
        )

    monkeypatch.setattr(
        gate0_module.AppWorldEpisodeExecutor,
        "execute_plan",
        unsuccessful_execute,
    )

    with pytest.raises(ValueError, match="paired AppWorld evidence failed") as raised:
        gate0_module._admit_variant_pair(
            source_adapter=source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=_WorldFactory(),
        )

    assert raised.value.diagnostic_codes == (  # type: ignore[attr-defined]
        diagnostic_code,
    )
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("kind", ["rename", "restructure"])
def test_transformed_candidate_must_be_bound_to_exact_reference_source(kind: str) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    reference_source, plan = _source_and_calls()
    _, candidate, _, translate = _candidate(kind)
    assert candidate.transform.source_variant == reference_source.variant
    assert candidate.transform.source_variant is not reference_source.variant
    factory = _WorldFactory()

    with pytest.raises(ValueError, match="trusted canonical translator") as raised:
        gate0_module._admit_variant_pair(
            source_adapter=reference_source,
            candidate_adapter=candidate,
            oracle_plan=plan,
            canonical_call_to_surface=translate,
            world_context_factory=factory,
        )

    assert raised.value.__cause__ is None
    assert factory.open_roles == []


def test_smoke_screens_in_loader_order_selects_l1_l2_and_freezes_world_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    ineligible_catalog = [
        _tool(
            "notes__search_notes",
            {"query": {"type": "string", "minLength": 1}},
            ["query"],
        )
    ]
    task_specs = {
        "PRIVATE_TASK_Z": (
            ineligible_catalog,
            ({"name": "notes__search_notes", "arguments": {"query": "first"}},),
        ),
        "PRIVATE_TASK_A": (
            _catalog(),
            (
                {
                    "name": "notes__create_note",
                    "arguments": {"title": "second", "priority": 1},
                },
            ),
        ),
    }
    factory = _SmokeContextFactory(task_specs)
    loader_calls: list[str] = []
    pin_calls = 0
    admitted: list[tuple[object, Callable[..., object]]] = []

    def task_loader(split: str) -> tuple[str, ...]:
        loader_calls.append(split)
        return ("PRIVATE_TASK_Z", "PRIVATE_TASK_A")

    def pin_checker() -> None:
        nonlocal pin_calls
        pin_calls += 1

    def admit_spy(**kwargs: object) -> object:
        candidate = kwargs["candidate_adapter"]
        pair_factory = kwargs["world_context_factory"]
        assert callable(pair_factory)
        admitted.append((candidate, pair_factory))
        for role in ("reference", "candidate", "reference-reset", "candidate-reset"):
            with pair_factory(role=role):
                pass
        return gate0_module._VariantAdmissionRecord(1, (("schema", 1),), ())

    monkeypatch.setattr(gate0_module, "_admit_variant_pair", admit_spy)

    summary = gate0_module.run_appworld_gate0_smoke(
        seed=100,
        workers=1,
        task_loader=task_loader,
        world_context_factory=factory,
        pin_checker=pin_checker,
    )

    assert loader_calls == ["train"]
    assert pin_calls == 1
    assert len(admitted) == 3
    assert admitted[0][0].variant.variant_id == "appworld-source-v1"
    assert admitted[1][0].transform.canonical_to_surface == {
        "notes__create_note": "toolshift_l1_0000",
        "notes__search_notes": "notes__search_notes",
    }
    assert admitted[2][0].transform.rules[0].tool_name == "notes__create_note"
    assert admitted[2][0].transform.rules[0].moved_parameters == ("priority",)
    assert len(admitted[2][0].transform.rules[0].moved_parameters) < 2
    assert summary.screened_task_count == 2
    assert summary.excluded_task_count == 1
    assert summary.admitted_task_count == 1
    assert summary.clean.admitted_count == 1
    assert summary.l1.admitted_count == 1
    assert summary.l2.admitted_count == 1
    assert summary.smoke_passed is True
    assert summary.to_dict()["smoke_passed"] is True
    assert all(counts["admitted"] == 1 for counts in summary.to_dict()["transform_counts"].values())
    assert "PRIVATE_TASK" not in repr(summary)
    assert factory.max_live_count == 1
    assert [call["task_id"] for call in factory.calls[:2]] == [
        "PRIVATE_TASK_Z",
        "PRIVATE_TASK_A",
    ]
    assert len(factory.calls) == 14
    assert all(call["task_id"] == "PRIVATE_TASK_A" for call in factory.calls[2:])
    common_flags = {
        "random_seed": 100,
        "raise_on_failure": False,
        "raise_on_extra_parameters": True,
        "remote_apis_url": None,
        "remote_environment_url": None,
        "remote_mcp_url": None,
        "remote_docker": False,
        "parse_datetimes": False,
        "wrap_response": False,
        "unwrap_response": False,
        "munchify_response": False,
    }
    experiment_names: list[str] = []
    for index, call in enumerate(factory.calls):
        assert {key: call[key] for key in common_flags} == common_flags
        assert call.get("ground_truth_mode") == ("full" if index in {0, 1} else None)
        experiment_name = call["experiment_name"]
        assert isinstance(experiment_name, str)
        experiment_names.append(experiment_name)
    assert len(experiment_names) == len(set(experiment_names))
    pair_profiles = tuple(
        {key: value for key, value in call.items() if key != "experiment_name"}
        for call in factory.calls[2:]
    )
    assert pair_profiles and all(profile == pair_profiles[0] for profile in pair_profiles)


def test_smoke_preflight_failure_is_static_and_precedes_task_loading() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    loader_calls: list[str] = []

    def failing_pin() -> None:
        raise RuntimeError("PRIVATE_PIN_PATH_CANARY")

    with pytest.raises(RuntimeError, match="AppWorld Gate 0 preflight failed") as raised:
        gate0_module.run_appworld_gate0_smoke(
            task_loader=lambda split: loader_calls.append(split) or (),
            world_context_factory=lambda **kwargs: pytest.fail("world construction must not occur"),
            pin_checker=failing_pin,
        )

    assert loader_calls == []
    assert raised.value.__cause__ is None
    assert "PRIVATE_PIN_PATH_CANARY" not in str(raised.value)


def test_smoke_without_selective_l2_is_a_failed_nonformal_summary() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    task_specs = {
        "PRIVATE_ONLY_TASK": (
            [
                _tool(
                    "notes__search_notes",
                    {"query": {"type": "string", "minLength": 1}},
                    ["query"],
                )
            ],
            ({"name": "notes__search_notes", "arguments": {"query": "one"}},),
        )
    }
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: ("PRIVATE_ONLY_TASK",),
        world_context_factory=_SmokeContextFactory(task_specs),
        pin_checker=lambda: None,
    )

    assert summary.smoke_passed is False
    assert summary.gate_evaluable is False
    assert summary.formal_gate_passed is False
    assert summary.screened_task_count == 1
    assert summary.l2.eligible_count == 0
    assert summary.l2.attempted_count == 0
    assert summary.diagnostic_counts == (("smoke.no_l2_eligible_task", 1),)


def test_first_eligible_task_admission_failure_does_not_select_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    task_specs = {
        task_id: (
            _catalog(),
            (
                {
                    "name": "notes__create_note",
                    "arguments": {"title": "one", "priority": 1},
                },
            ),
        )
        for task_id in ("PRIVATE_FIRST", "PRIVATE_SECOND")
    }
    factory = _SmokeContextFactory(task_specs)

    def reject_first_pair(**kwargs: object) -> object:
        del kwargs
        raise gate0_module._VariantAdmissionError(
            "paired AppWorld evidence failed",
            ("pair.synthetic_failure",),
        )

    monkeypatch.setattr(gate0_module, "_admit_variant_pair", reject_first_pair)
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: ("PRIVATE_FIRST", "PRIVATE_SECOND"),
        world_context_factory=factory,
        pin_checker=lambda: None,
    )

    assert summary.smoke_passed is False
    assert summary.screened_task_count == 1
    assert summary.clean.attempted_count == 1
    assert summary.l1.attempted_count == 0
    assert summary.diagnostic_counts == (("pair.unsafe_diagnostic", 1),)
    assert [call["task_id"] for call in factory.calls] == ["PRIVATE_FIRST"]


class _ReusingSmokeContextFactory(_SmokeContextFactory):
    def __init__(
        self,
        task_specs: Mapping[
            str,
            tuple[list[dict[str, object]], tuple[Mapping[str, object], ...]],
        ],
    ) -> None:
        super().__init__(task_specs)
        self._shared_world: _SmokeWorld | None = None

    @contextmanager
    def __call__(self, **kwargs: object) -> Iterator[_SmokeWorld]:
        self.calls.append(dict(kwargs))
        task_id = kwargs["task_id"]
        assert isinstance(task_id, str)
        catalog, oracle_calls = self.task_specs[task_id]
        if self._shared_world is None:
            self._shared_world = _SmokeWorld(catalog, oracle_calls, self.events)
        self.live_count += 1
        self.max_live_count = max(self.max_live_count, self.live_count)
        try:
            yield self._shared_world
        finally:
            self.live_count -= 1


def test_smoke_rejects_world_identity_reuse_across_variant_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    task_specs = {
        "PRIVATE_REUSED": (
            _catalog(),
            (
                {
                    "name": "notes__create_note",
                    "arguments": {"title": "one", "priority": 1},
                },
            ),
        )
    }
    factory = _ReusingSmokeContextFactory(task_specs)

    def open_reference(**kwargs: object) -> object:
        pair_factory = kwargs["world_context_factory"]
        assert callable(pair_factory)
        with pair_factory(role="reference"):
            pass
        return gate0_module._VariantAdmissionRecord(1, (("schema", 1),), ())

    monkeypatch.setattr(gate0_module, "_admit_variant_pair", open_reference)
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: ("PRIVATE_REUSED",),
        world_context_factory=factory,
        pin_checker=lambda: None,
    )

    assert summary.smoke_passed is False
    assert summary.execution_exception_count == 1
    assert summary.diagnostic_counts == (("smoke.admission_exception", 1),)
    assert len(factory.calls) == 2


class _CleanupMarkedBodyFailureFactory(_SmokeContextFactory):
    def __call__(self, **kwargs: object) -> object:
        inner = super().__call__(**kwargs)

        class CleanupMarkedContext:
            def __enter__(self) -> object:
                return inner.__enter__()

            def __exit__(self, *exception: object) -> bool | None:
                error = exception[1]
                if isinstance(error, BaseException):
                    from toolshift.benchmarks.appworld_replay import _AppWorldCleanupMarker

                    error.__cause__ = _AppWorldCleanupMarker(
                        "instance_close",
                        "RuntimeError",
                        fatal=False,
                    )
                return inner.__exit__(*exception)

        return CleanupMarkedContext()


def test_smoke_preserves_pair_diagnostic_with_body_and_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    task_specs = {
        "PRIVATE_DUAL_FAILURE": (
            _catalog(),
            (
                {
                    "name": "notes__create_note",
                    "arguments": {"title": "one", "priority": 1},
                },
            ),
        )
    }

    def fail_reset_hash(**kwargs: object) -> object:
        pair_factory = kwargs["world_context_factory"]
        assert callable(pair_factory)
        with pair_factory(role="reference"):
            raise gate0_module._VariantAdmissionError(
                "paired AppWorld execution failed",
                ("pair.reset_hash_exception",),
            )

    monkeypatch.setattr(gate0_module, "_admit_variant_pair", fail_reset_hash)
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: ("PRIVATE_DUAL_FAILURE",),
        world_context_factory=_CleanupMarkedBodyFailureFactory(task_specs),
        pin_checker=lambda: None,
    )

    assert summary.diagnostic_counts == (("pair.reset_hash_exception", 1),)
    assert summary.execution_exception_count == 1
    assert summary.cleanup_exception_count == 1
    assert summary.smoke_passed is False


class _CleanupFailingSmokeContextFactory(_SmokeContextFactory):
    @contextmanager
    def __call__(self, **kwargs: object) -> Iterator[_SmokeWorld]:
        with super().__call__(**kwargs) as world:
            yield world
        raise RuntimeError("PRIVATE_CLEANUP_PATH_CANARY")


def test_smoke_cleanup_failure_is_counted_and_payload_free() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    task_specs = {
        "PRIVATE_CLEANUP": (
            _catalog(),
            (
                {
                    "name": "notes__create_note",
                    "arguments": {"title": "one", "priority": 1},
                },
            ),
        )
    }
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: ("PRIVATE_CLEANUP",),
        world_context_factory=_CleanupFailingSmokeContextFactory(task_specs),
        pin_checker=lambda: None,
    )

    assert summary.smoke_passed is False
    assert summary.cleanup_exception_count == 1
    assert summary.execution_exception_count == 0
    assert summary.diagnostic_counts == (("smoke.world_lifecycle_failure", 1),)
    assert "PRIVATE" not in repr(summary)


class _ConstructorAndCleanupFailingContextFactory:
    @contextmanager
    def __call__(self, **kwargs: object) -> Iterator[object]:
        del kwargs
        from toolshift.benchmarks.appworld_replay import _AppWorldCleanupMarker

        marker = _AppWorldCleanupMarker(
            "constructor_close_all",
            "RuntimeError",
            fatal=False,
        )
        raise RuntimeError("PRIVATE_CONSTRUCTOR_CANARY") from marker
        yield  # pragma: no cover


def test_constructor_failure_with_cleanup_marker_counts_both_failures() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: ("PRIVATE_CONSTRUCTOR",),
        world_context_factory=_ConstructorAndCleanupFailingContextFactory(),
        pin_checker=lambda: None,
    )

    assert summary.smoke_passed is False
    assert summary.execution_exception_count == 1
    assert summary.cleanup_exception_count == 1
    assert summary.diagnostic_counts == (("smoke.world_lifecycle_failure", 1),)
    assert "PRIVATE" not in repr(summary)


class _ExitReentrantContextFactory:
    def __init__(self) -> None:
        self.contexts: object | None = None
        self.call_count = 0
        self.live_count = 0
        self.max_live_count = 0
        self.reentered = False

    @contextmanager
    def __call__(self, **kwargs: object) -> Iterator[object]:
        del kwargs
        self.call_count += 1
        self.live_count += 1
        self.max_live_count = max(self.max_live_count, self.live_count)
        try:
            yield object()
            if not self.reentered:
                self.reentered = True
                assert self.contexts is not None
                with self.contexts.open(task_id="PRIVATE_REENTRY", role="candidate"):
                    pass
        finally:
            self.live_count -= 1


def test_world_context_exit_cannot_reenter_before_live_release() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    factory = _ExitReentrantContextFactory()
    accounting = gate0_module._SmokeAccounting()
    contexts = gate0_module._SmokeWorldContexts(factory, seed=100, accounting=accounting)
    factory.contexts = contexts

    with (
        pytest.raises(ValueError, match="world lifecycle failed") as raised,
        contexts.open(task_id="PRIVATE_REENTRY", role="screen"),
    ):
        pass

    assert raised.value.__cause__ is None
    assert factory.call_count == 1
    assert factory.max_live_count == 1


class _SingleWorldContextFactory:
    def __init__(self) -> None:
        self.call_count = 0

    @contextmanager
    def __call__(self, **kwargs: object) -> Iterator[object]:
        del kwargs
        self.call_count += 1
        yield object()


def test_world_context_live_slot_is_process_global_across_managers() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    first_accounting = gate0_module._SmokeAccounting()
    second_accounting = gate0_module._SmokeAccounting()
    first_factory = _SingleWorldContextFactory()
    second_factory = _SingleWorldContextFactory()
    first = gate0_module._SmokeWorldContexts(
        first_factory,
        seed=100,
        accounting=first_accounting,
    )
    second = gate0_module._SmokeWorldContexts(
        second_factory,
        seed=100,
        accounting=second_accounting,
    )

    with (
        first.open(task_id="PRIVATE_FIRST", role="screen"),
        pytest.raises(ValueError, match="world lifecycle failed"),
        second.open(task_id="PRIVATE_SECOND", role="screen"),
    ):
        pass

    assert first_factory.call_count == 1
    assert second_factory.call_count == 0
    assert first_accounting.execution_exception_count == 0
    assert second_accounting.execution_exception_count == 1


class _ReuseWithCleanupFailureFactory:
    def __init__(self) -> None:
        self.world = object()
        self.call_count = 0

    @contextmanager
    def __call__(self, **kwargs: object) -> Iterator[object]:
        del kwargs
        self.call_count += 1
        try:
            yield self.world
        finally:
            if self.call_count == 2:
                from toolshift.benchmarks.appworld_replay import _AppWorldCleanupMarker

                marker = _AppWorldCleanupMarker(
                    "instance_close",
                    "RuntimeError",
                    fatal=False,
                )
                raise RuntimeError("PRIVATE_CLEANUP_CANARY") from marker


def test_world_reuse_plus_cleanup_failure_counts_both_boundaries() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    factory = _ReuseWithCleanupFailureFactory()
    accounting = gate0_module._SmokeAccounting()
    contexts = gate0_module._SmokeWorldContexts(factory, seed=100, accounting=accounting)

    with contexts.open(task_id="PRIVATE_REUSE", role="screen"):
        pass
    with (
        pytest.raises(RuntimeError, match="PRIVATE_CLEANUP_CANARY"),
        contexts.open(
            task_id="PRIVATE_REUSE",
            role="screen",
        ),
    ):
        pass

    assert accounting.execution_exception_count == 1
    assert accounting.cleanup_exception_count == 1


class _SuppressingContextFactory:
    def __call__(self, **kwargs: object) -> object:
        del kwargs

        class SuppressingContext:
            def __enter__(self) -> object:
                return object()

            def __exit__(self, *exception: object) -> bool:
                del exception
                return True

        return SuppressingContext()


def test_world_factory_cannot_suppress_episode_body_failure() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    accounting = gate0_module._SmokeAccounting()
    contexts = gate0_module._SmokeWorldContexts(
        _SuppressingContextFactory(),
        seed=100,
        accounting=accounting,
    )

    with (
        pytest.raises(ValueError, match="world lifecycle failed") as raised,
        contexts.open(
            task_id="PRIVATE_SUPPRESSED",
            role="screen",
        ),
    ):
        raise RuntimeError("PRIVATE_BODY_CANARY")

    assert raised.value.__cause__ is None
    assert accounting.execution_exception_count == 1
    assert accounting.cleanup_exception_count == 0


def test_private_appworld_root_requires_absolute_owned_0700_non_git_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private-root"
    private_root.mkdir(mode=0o700)
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))

    assert gate0_module._require_private_appworld_root() == private_root

    private_root.chmod(0o755)
    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_private_appworld_root()
    assert str(private_root) not in str(raised.value)
    assert raised.value.__cause__ is None


def _write_private_appworld_versions(root: Path) -> None:
    data = root / "data"
    base_dbs = data / "base_dbs"
    base_dbs.mkdir(parents=True, mode=0o700)
    data.chmod(0o700)
    base_dbs.chmod(0o700)
    (data / "version.txt").write_text("0.2.0\n", encoding="utf-8")
    (base_dbs / "version.txt").write_text("0.2.0\n", encoding="utf-8")
    (data / "version.txt").chmod(0o600)
    (base_dbs / "version.txt").chmod(0o600)


def test_private_appworld_tree_precreates_private_runtime_directories(
    tmp_path: Path,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private-root"
    private_root.mkdir(mode=0o700)
    _write_private_appworld_versions(private_root)

    gate0_module._require_private_appworld_tree(private_root)

    for relative in (
        "experiments",
        "experiments/outputs",
        ".cache",
        ".tmp",
        ".profiling",
        "plots",
        ".release",
    ):
        directory = private_root / relative
        assert directory.is_dir()
        assert not directory.is_symlink()
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_private_appworld_tree_creation_is_umask_independent(tmp_path: Path) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private-root"
    private_root.mkdir(mode=0o700)
    _write_private_appworld_versions(private_root)
    previous_umask = os.umask(0o777)
    try:
        gate0_module._require_private_appworld_tree(private_root)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((private_root / "experiments").stat().st_mode) == 0o700
    assert stat.S_IMODE((private_root / "experiments" / "outputs").stat().st_mode) == 0o700


def test_private_appworld_directory_closes_descriptor_when_parent_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    original_open = os.open
    original_close = os.close
    opened: list[int] = []
    closed: list[int] = []
    parent_fd = original_open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)

    def tracking_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    def fail_fsync(descriptor: int) -> None:
        assert descriptor == parent_fd
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(gate0_module.os, "open", tracking_open)
    monkeypatch.setattr(gate0_module.os, "close", tracking_close)
    monkeypatch.setattr(gate0_module.os, "fsync", fail_fsync)
    try:
        with pytest.raises(OSError, match="synthetic fsync failure"):
            gate0_module._open_private_appworld_directory(
                "created",
                parent_fd=parent_fd,
                create=True,
            )
    finally:
        original_close(parent_fd)

    assert len(opened) == 1
    assert opened[0] in closed


def test_private_evidence_directory_closes_descriptor_when_normalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks._evidence as evidence_module

    child = tmp_path / "child"
    child.mkdir(mode=0o700)
    original_open = os.open
    original_close = os.close
    opened: list[int] = []
    closed: list[int] = []
    parent_fd = original_open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)

    def tracking_open(*args: object, **kwargs: object) -> int:
        descriptor = original_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    def tracking_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    def fail_fchmod(descriptor: int, mode: int) -> None:
        assert descriptor == opened[-1]
        assert mode == 0o700
        raise OSError("synthetic chmod failure")

    monkeypatch.setattr(evidence_module.os, "open", tracking_open)
    monkeypatch.setattr(evidence_module.os, "close", tracking_close)
    monkeypatch.setattr(evidence_module.os, "fchmod", fail_fchmod)
    try:
        with pytest.raises(OSError, match="synthetic chmod failure"):
            evidence_module._open_private_directory(
                "child",
                parent_fd=parent_fd,
                normalize_mode=True,
            )
    finally:
        original_close(parent_fd)

    assert len(opened) == 1
    assert opened[0] in closed


@pytest.mark.parametrize(
    "relative",
    ["data", "data/base_dbs", "experiments", "experiments/outputs"],
)
def test_private_appworld_tree_rejects_protected_subtree_symlinks(
    tmp_path: Path,
    relative: str,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private-root"
    private_root.mkdir(mode=0o700)
    _write_private_appworld_versions(private_root)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    target = private_root / relative
    if target.exists():
        if target.is_dir():
            for child in sorted(target.rglob("*"), reverse=True):
                child.unlink() if child.is_file() else child.rmdir()
            target.rmdir()
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.parent.chmod(0o700)
    target.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_private_appworld_tree(private_root)
    assert str(outside) not in str(raised.value)
    assert raised.value.__cause__ is None


def test_editable_checkout_pin_uses_direct_url_and_exact_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()

    class Distribution:
        version = "0.2.0.dev0"

        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "direct_url.json"
            return f'{{"dir_info":{{"editable":true}},"url":"{checkout.as_uri()}"}}'

    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )
    monkeypatch.setenv("GIT_DIR", "PRIVATE_REDIRECTED_GIT_DIR")
    monkeypatch.setenv("GIT_WORK_TREE", "PRIVATE_REDIRECTED_WORK_TREE")
    monkeypatch.setenv("GIT_INDEX_FILE", "PRIVATE_REDIRECTED_INDEX")
    attached = False
    dirty = False

    def git_run(arguments: list[str], **kwargs: object) -> object:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        git_environment_keys = {key for key in environment if key.startswith("GIT_")}
        assert not git_environment_keys
        assert f"--git-dir={checkout / '.git'}" in arguments
        assert f"--work-tree={checkout}" in arguments
        if "--show-toplevel" in arguments:
            return SimpleNamespace(
                stdout=f"{checkout}\n{checkout / '.git'}\n",
                returncode=0,
            )
        if "--show-object-format" in arguments:
            return SimpleNamespace(stdout="sha1\n", returncode=0)
        if "rev-parse" in arguments:
            return SimpleNamespace(
                stdout=gate0_module.PINNED_APPWORLD_COMMIT + "\n",
                returncode=0,
            )
        if "symbolic-ref" in arguments:
            return SimpleNamespace(stdout="", returncode=0 if attached else 1)
        if "--others" in arguments:
            return SimpleNamespace(stdout=b"", returncode=0)
        if "ls-files" in arguments:
            return SimpleNamespace(stdout="H tracked.py\0", returncode=0)
        assert "diff-index" in arguments
        return SimpleNamespace(stdout="", returncode=0)

    def require_raw_match(
        raw_checkout: Path,
        git_prefix: list[str],
        environment: dict[str, str],
    ) -> None:
        assert raw_checkout == checkout
        assert f"--git-dir={checkout / '.git'}" in git_prefix
        assert not {key for key in environment if key.startswith("GIT_")}
        if dirty:
            raise ValueError

    monkeypatch.setattr(gate0_module.subprocess, "run", git_run)
    monkeypatch.setattr(gate0_module, "_require_head_worktree_match", require_raw_match)

    assert gate0_module._require_editable_appworld_checkout() == checkout

    attached = True
    with pytest.raises(RuntimeError, match="preflight failed"):
        gate0_module._require_editable_appworld_checkout()
    attached = False

    dirty = True
    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_editable_appworld_checkout()
    assert raised.value.__cause__ is None
    dirty = False

    Distribution.version = "PRIVATE_WRONG_VERSION"
    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_editable_appworld_checkout()
    assert "PRIVATE_WRONG_VERSION" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_editable_checkout_pin_ignores_local_core_worktree_redirect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    checkout = tmp_path / "checkout"
    shadow = tmp_path / "shadow"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    tracked = checkout / "tracked.py"
    tracked.write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "synthetic"], check=True)
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "clone", "-q", str(checkout), str(shadow)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "-q"], check=True)
    subprocess.run(
        [
            "git",
            f"--git-dir={checkout / '.git'}",
            "config",
            "core.worktree",
            str(shadow),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "update-index", "--refresh"], check=True)
    tracked.write_text("PINNED = False\n", encoding="utf-8")
    redirected_top_level = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert Path(redirected_top_level).resolve() == shadow.resolve()
    assert (
        subprocess.run(
            ["git", "-C", str(checkout), "diff-index", "--quiet", "HEAD", "--"],
            check=False,
        ).returncode
        == 0
    )

    class Distribution:
        version = "0.2.0.dev0"

        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "direct_url.json"
            return f'{{"dir_info":{{"editable":true}},"url":"{checkout.as_uri()}"}}'

    monkeypatch.setattr(gate0_module, "PINNED_APPWORLD_COMMIT", head)
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_editable_appworld_checkout()
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_editable_checkout_pin_rejects_tracked_changes_hidden_by_index_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_flag: str,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    tracked = checkout / "tracked.py"
    tracked.write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "synthetic"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "-q"], check=True)
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(checkout), "update-index", index_flag, "tracked.py"],
        check=True,
    )
    tracked.write_text("PINNED = False\n", encoding="utf-8")
    assert (
        subprocess.run(
            ["git", "-C", str(checkout), "diff-index", "--quiet", "HEAD", "--"],
            check=False,
        ).returncode
        == 0
    )

    class Distribution:
        version = "0.2.0.dev0"

        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "direct_url.json"
            return f'{{"dir_info":{{"editable":true}},"url":"{checkout.as_uri()}"}}'

    monkeypatch.setattr(gate0_module, "PINNED_APPWORLD_COMMIT", head)
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_editable_appworld_checkout()
    assert raised.value.__cause__ is None


def test_editable_checkout_pin_accepts_real_clean_detached_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    (checkout / "tracked.py").write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "synthetic"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "-q"], check=True)
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    class Distribution:
        version = "0.2.0.dev0"

        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "direct_url.json"
            return f'{{"dir_info":{{"editable":true}},"url":"{checkout.as_uri()}"}}'

    monkeypatch.setattr(gate0_module, "PINNED_APPWORLD_COMMIT", head)
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    assert gate0_module._require_editable_appworld_checkout() == checkout.resolve()


def test_editable_checkout_pin_rejects_staged_only_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    tracked = checkout / "tracked.py"
    tracked.write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "synthetic"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "-q"], check=True)
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    tracked.write_text("PINNED = False\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
    tracked.write_text("PINNED = True\n", encoding="utf-8")
    assert (
        subprocess.run(
            ["git", "-C", str(checkout), "diff", "--cached", "--quiet", "HEAD", "--"],
            check=False,
        ).returncode
        == 1
    )

    class Distribution:
        version = "0.2.0.dev0"

        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "direct_url.json"
            return f'{{"dir_info":{{"editable":true}},"url":"{checkout.as_uri()}"}}'

    monkeypatch.setattr(gate0_module, "PINNED_APPWORLD_COMMIT", head)
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_editable_appworld_checkout()
    assert raised.value.__cause__ is None


def test_editable_checkout_pin_rejects_local_attribute_filter_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    tracked = checkout / "tracked.py"
    tracked.write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "synthetic"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "-q"], check=True)
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    (checkout / ".git" / "info" / "attributes").write_text(
        "tracked.py filter=synthetic-mask\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "config",
            "filter.synthetic-mask.clean",
            "sed 's/False/True/'",
        ],
        check=True,
    )
    tracked.write_text("PINNED = False\n", encoding="utf-8")
    assert (
        subprocess.run(
            ["git", "-C", str(checkout), "diff", "--quiet", "--", "tracked.py"],
            check=False,
        ).returncode
        == 0
    )

    class Distribution:
        version = "0.2.0.dev0"

        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "direct_url.json"
            return f'{{"dir_info":{{"editable":true}},"url":"{checkout.as_uri()}"}}'

    monkeypatch.setattr(gate0_module, "PINNED_APPWORLD_COMMIT", head)
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_editable_appworld_checkout()
    assert raised.value.__cause__ is None


def test_editable_checkout_pin_verifies_expanded_lfs_payload_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    payload = b"synthetic expanded large-file payload\n"
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        + b"oid sha256:"
        + hashlib.sha256(payload).hexdigest().encode("ascii")
        + b"\nsize "
        + str(len(payload)).encode("ascii")
        + b"\n"
    )
    tracked = checkout / "tracked.bundle"
    tracked.write_bytes(pointer)
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.bundle"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "synthetic"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "-q"], check=True)
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.write_bytes(payload)

    class Distribution:
        version = "0.2.0.dev0"

        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "direct_url.json"
            return f'{{"dir_info":{{"editable":true}},"url":"{checkout.as_uri()}"}}'

    monkeypatch.setattr(gate0_module, "PINNED_APPWORLD_COMMIT", head)
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    assert gate0_module._require_editable_appworld_checkout() == checkout.resolve()

    tracked.write_bytes(payload + b"tampered")
    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_editable_appworld_checkout()
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    "relative",
    [
        "src/appworld/untracked_runtime.py",
        "src/appworld/apps/admin/untracked_runtime.py",
        ".env",
    ],
)
def test_editable_checkout_pin_rejects_untracked_runtime_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    package = checkout / "src" / "appworld"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("PINNED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "src/appworld/__init__.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "synthetic"], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "-q"], check=True)
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    untracked = checkout / relative
    untracked.parent.mkdir(parents=True, exist_ok=True)
    untracked.write_text(
        "PRIVATE_RUNTIME_OVERRIDE = True\n",
        encoding="utf-8",
    )

    class Distribution:
        version = "0.2.0.dev0"

        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "direct_url.json"
            return f'{{"dir_info":{{"editable":true}},"url":"{checkout.as_uri()}"}}'

    monkeypatch.setattr(gate0_module, "PINNED_APPWORLD_COMMIT", head)
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "distribution",
        lambda name: Distribution(),
    )

    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module._require_editable_appworld_checkout()
    assert "PRIVATE_RUNTIME_OVERRIDE" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_pinned_runtime_checks_proxy_python_constants_and_version_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private-root"
    private_root.mkdir(mode=0o700)
    _write_private_appworld_versions(private_root)
    for key in tuple(os.environ):
        if key.lower().endswith("_proxy"):
            monkeypatch.delenv(key, raising=False)
    for key in ("APPWORLD_DB_ARGS", "APPWORLD_DATE_TIME", "LOAD_ON_STARTUP"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    monkeypatch.setenv("APPWORLD_CACHE", str(private_root / ".cache"))
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(gate0_module.sys, "version_info", (3, 11, 15, "final", 0))
    monkeypatch.setattr(gate0_module.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "version",
        lambda name: "1.2.2" if name == "python-dotenv" else "unexpected",
    )
    monkeypatch.setattr(
        gate0_module,
        "_require_private_appworld_root",
        lambda: private_root,
    )
    checkout = tmp_path / "checkout"
    package_file = checkout / "src" / "appworld" / "__init__.py"
    constants_file = checkout / "src" / "appworld" / "common" / "constants.py"
    constants_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="utf-8")
    constants_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        gate0_module,
        "_require_editable_appworld_checkout",
        lambda: checkout,
    )
    appworld_module = ModuleType("appworld")
    common_module = ModuleType("appworld.common")
    constants_module = ModuleType("appworld.common.constants")
    appworld_module.__file__ = str(package_file)
    constants_module.__file__ = str(constants_file)
    constants_module.DATA_VERSION = "0.2.0"  # type: ignore[attr-defined]
    constants_module.DB_VERSION = "0.2.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "appworld", appworld_module)
    monkeypatch.setitem(sys.modules, "appworld.common", common_module)
    monkeypatch.setitem(sys.modules, "appworld.common.constants", constants_module)

    dotenv_module = ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda *, stream, override: False  # type: ignore[attr-defined]
    original_import = gate0_module.importlib.import_module

    def import_runtime_module(name: str) -> ModuleType:
        if name == "dotenv":
            return dotenv_module
        return original_import(name)

    monkeypatch.setattr(gate0_module.importlib, "import_module", import_runtime_module)

    gate0_module.require_pinned_appworld_runtime()

    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "version",
        lambda name: "1.1.1" if name == "python-dotenv" else "unexpected",
    )
    with pytest.raises(RuntimeError, match="preflight failed"):
        gate0_module.require_pinned_appworld_runtime()
    monkeypatch.setattr(
        gate0_module.importlib.metadata,
        "version",
        lambda name: "1.2.2" if name == "python-dotenv" else "unexpected",
    )

    def mutate_from_dotenv(*, stream: object, override: bool) -> bool:
        del stream, override
        monkeypatch.setenv("TOOLSHIFT_DOTENV_DISABLED_PROBE", "mutated")
        return True

    monkeypatch.setattr(dotenv_module, "load_dotenv", mutate_from_dotenv, raising=False)
    with pytest.raises(RuntimeError, match="preflight failed"):
        gate0_module.require_pinned_appworld_runtime()
    assert "TOOLSHIFT_DOTENV_DISABLED_PROBE" not in os.environ
    monkeypatch.setattr(
        dotenv_module,
        "load_dotenv",
        lambda *, stream, override: False,
        raising=False,
    )

    shadow_file = tmp_path / "shadow" / "appworld" / "__init__.py"
    shadow_file.parent.mkdir(parents=True)
    shadow_file.write_text("", encoding="utf-8")
    appworld_module.__file__ = str(shadow_file)
    with pytest.raises(RuntimeError, match="preflight failed"):
        gate0_module.require_pinned_appworld_runtime()
    appworld_module.__file__ = str(package_file)

    monkeypatch.setenv("CUSTOM_PROXY", "PRIVATE_PROXY_CANARY")
    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module.require_pinned_appworld_runtime()
    assert "PRIVATE_PROXY_CANARY" not in str(raised.value)
    assert raised.value.__cause__ is None
    monkeypatch.delenv("CUSTOM_PROXY")

    original_import = gate0_module.importlib.import_module

    def import_with_environment_mutation(name: str) -> ModuleType:
        module = original_import(name)
        if name == "appworld.common.constants":
            monkeypatch.delenv("PYTHONDONTWRITEBYTECODE")
        return module

    monkeypatch.setattr(gate0_module.importlib, "import_module", import_with_environment_mutation)
    with pytest.raises(RuntimeError, match="preflight failed") as raised:
        gate0_module.require_pinned_appworld_runtime()
    assert raised.value.__cause__ is None


def test_empty_oracle_fails_the_smoke_without_trying_a_later_task() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    task_specs = {
        "PRIVATE_EMPTY": (_catalog(), ()),
        "PRIVATE_LATER": (
            _catalog(),
            (
                {
                    "name": "notes__create_note",
                    "arguments": {"title": "later", "priority": 1},
                },
            ),
        ),
    }
    factory = _SmokeContextFactory(task_specs)
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: ("PRIVATE_EMPTY", "PRIVATE_LATER"),
        world_context_factory=factory,
        pin_checker=lambda: None,
    )

    assert summary.smoke_passed is False
    assert summary.screened_task_count == 1
    assert summary.diagnostic_counts == (("screen.oracle_capture_failed", 1),)
    assert [call["task_id"] for call in factory.calls] == ["PRIVATE_EMPTY"]


def test_l1_l2_selection_is_deterministic_selective_and_nonmutating() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source = build_appworld_adapter(_catalog())
    source_schema = next(
        tool.input_schema for tool in source.variant.tools if tool.name == "notes__create_note"
    )
    before = canonical_json_bytes(source_schema)
    oracle_tools = ("notes__create_note",)

    l1 = gate0_module._build_l1_candidate(source, oracle_tools, seed=100)
    l2 = gate0_module._build_l2_candidate(source, oracle_tools, seed=100)

    assert l1.transform.canonical_to_surface["notes__create_note"] == ("toolshift_l1_0000")
    assert l2 is not None
    assert l2.transform.rules[0].container_name == "toolshift_group_0000"
    assert l2.transform.rules[0].moved_parameters == ("priority",)
    assert len(l2.transform.rules[0].moved_parameters) < len(source_schema["properties"])
    assert canonical_json_bytes(source_schema) == before


def test_l2_does_not_delete_defaults_to_manufacture_eligibility() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    source = build_appworld_adapter(
        [
            _tool(
                "notes__create_note",
                {
                    "priority": {"type": "integer", "default": 1},
                    "title": {"type": "string", "minLength": 1},
                },
                ["title"],
            )
        ]
    )
    source_schema = source.variant.tools[0].input_schema
    before = canonical_json_bytes(source_schema)

    candidate = gate0_module._build_l2_candidate(
        source,
        ("notes__create_note",),
        seed=100,
    )

    assert candidate is None
    assert source_schema["properties"]["priority"]["default"] == 1
    assert canonical_json_bytes(source_schema) == before


def test_gate0_summary_serializes_only_the_exact_aggregate_allowlist() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )

    payload = summary.to_dict()
    assert set(payload) == {
        "schema_version",
        "mode",
        "pinned_appworld_commit",
        "pinned_appworld_package_version",
        "pinned_appworld_data_version",
        "pinned_appworld_base_db_version",
        "pinned_python_version",
        "adapter_abi_sha256",
        "runner_abi_sha256",
        "split",
        "seed",
        "workers",
        "screened_task_count",
        "admitted_task_count",
        "transform_counts",
        "diagnostic_counts",
        "execution_exception_count",
        "cleanup_exception_count",
        "gate_evaluable",
        "formal_gate_passed",
        "smoke_passed",
    }
    assert set(payload["transform_counts"]) == {"clean", "l1", "l2"}
    assert all(
        set(counts) == {"requested", "eligible", "attempted", "admitted", "excluded"}
        for counts in payload["transform_counts"].values()
    )
    assert payload["mode"] == "smoke"
    assert payload["pinned_appworld_commit"] == gate0_module.PINNED_APPWORLD_COMMIT
    assert payload["pinned_appworld_package_version"] == "0.2.0.dev0"
    assert payload["pinned_appworld_data_version"] == "0.2.0"
    assert payload["pinned_appworld_base_db_version"] == "0.2.0"
    assert payload["pinned_python_version"] == "3.11.15"
    assert re.fullmatch(r"[0-9a-f]{64}", payload["adapter_abi_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", payload["runner_abi_sha256"])
    assert payload["adapter_abi_sha256"] == (
        "424f7345df2845391065cce2004629a0a9839e7401684020e49ce1703ed5e61e"
    )
    assert payload["runner_abi_sha256"] == (
        "151ee5d1de762e31acfcea9f239e5ec6d4210ec39ce13b20910291ef889176cc"
    )
    assert payload["gate_evaluable"] is False
    assert payload["formal_gate_passed"] is False
    assert payload["smoke_passed"] is False
    assert "excluded_task_count" not in payload

    encoded = json.dumps(payload, sort_keys=True)
    forbidden_key_fragments = (
        "task_id",
        "schema_sha",
        "trace_sha",
        "state_sha",
        "suite",
        "fingerprint",
        "PRIVATE",
    )
    assert all(fragment not in encoded for fragment in forbidden_key_fragments)


def test_gate0_summary_writer_is_private_atomic_and_root_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    output = private_root / "reports" / "smoke.json"
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )

    gate0_module.write_appworld_gate0_summary(
        output,
        summary,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == summary.to_dict()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600

    output.chmod(0o644)
    gate0_module.write_appworld_gate0_summary(
        output,
        summary,
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o600

    restrictive_output = private_root / "restrictive" / "smoke.json"
    previous_umask = os.umask(0o777)
    try:
        gate0_module.write_appworld_gate0_summary(restrictive_output, summary)
    finally:
        os.umask(previous_umask)
    assert stat.S_IMODE(restrictive_output.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(restrictive_output.stat().st_mode) == 0o600

    with pytest.raises(ValueError, match="private AppWorld root") as raised:
        gate0_module.write_appworld_gate0_summary(
            tmp_path / "outside.json",
            summary,
        )
    assert str(tmp_path) not in str(raised.value)
    assert raised.value.__cause__ is None

    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    escape = private_root / "escape"
    escape.symlink_to(outside_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="private AppWorld root"):
        gate0_module.write_appworld_gate0_summary(
            escape / "escaped.json",
            summary,
        )
    assert not (outside_directory / "escaped.json").exists()


def test_gate0_summary_writer_cannot_escape_after_parent_binding_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks._evidence as evidence_module
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    reports = private_root / "reports"
    reports.mkdir(mode=0o700)
    displaced = private_root / "displaced"
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    real_replace = os.replace
    swapped = False

    def swap_parent_then_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if not swapped and src_dir_fd is not None and dst_dir_fd is not None:
            swapped = True
            os.rename(reports, displaced)
            reports.symlink_to(outside, target_is_directory=True)
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(evidence_module.os, "replace", swap_parent_then_replace)

    with pytest.raises(ValueError, match="evidence write failed") as raised:
        gate0_module.write_appworld_gate0_summary(reports / "smoke.json", summary)
    assert raised.value.__cause__ is None
    assert swapped is True
    assert not (outside / "smoke.json").exists()
    assert not (displaced / "smoke.json").exists()
    assert not tuple(displaced.glob("*.tmp"))


def test_gate0_summary_writer_keeps_installed_inode_when_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks._evidence as evidence_module
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    reports = private_root / "reports"
    reports.mkdir(mode=0o700)
    output = reports / "smoke.json"
    output.write_bytes(b"OLD")
    output.chmod(0o600)
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("PRIVATE_DIRECTORY_FSYNC_CANARY")
        real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "fsync", fail_directory_fsync)

    with pytest.raises(ValueError, match="evidence write failed") as raised:
        gate0_module.write_appworld_gate0_summary(output, summary)
    assert raised.value.__cause__ is None
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == summary.to_dict()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_gate0_summary_writer_rechecks_binding_after_durability_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks._evidence as evidence_module
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    reports = private_root / "reports"
    reports.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    stolen = outside / "stolen"
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    real_fsync = os.fsync
    moved = False

    def move_parent_during_directory_fsync(descriptor: int) -> None:
        nonlocal moved
        if not moved and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            moved = True
            os.rename(reports, stolen)
        real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "fsync", move_parent_during_directory_fsync)

    with pytest.raises(ValueError, match="evidence write failed") as raised:
        gate0_module.write_appworld_gate0_summary(reports / "smoke.json", summary)
    assert raised.value.__cause__ is None
    assert moved is True
    assert not (stolen / "smoke.json").exists()


def test_gate0_summary_writer_rechecks_private_root_mode_after_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import toolshift.benchmarks._evidence as evidence_module
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    reports = private_root / "reports"
    reports.mkdir(mode=0o700)
    output = reports / "smoke.json"
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    real_fsync = os.fsync
    changed = False

    def change_root_mode_during_directory_fsync(descriptor: int) -> None:
        nonlocal changed
        if not changed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            changed = True
            private_root.chmod(0o755)
        real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "fsync", change_root_mode_during_directory_fsync)

    with pytest.raises(ValueError, match="evidence write failed") as raised:
        gate0_module.write_appworld_gate0_summary(output, summary)
    assert raised.value.__cause__ is None
    assert changed is True
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["replace", "chmod"])
def test_gate0_summary_writer_rechecks_installed_leaf_after_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import toolshift.benchmarks._evidence as evidence_module
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    reports = private_root / "reports"
    reports.mkdir(mode=0o700)
    output = reports / "smoke.json"
    forged = reports / "forged.json"
    forged.write_bytes(b"FORGED")
    forged.chmod(0o644)
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    real_fsync = os.fsync
    mutated = False

    def mutate_leaf_during_directory_fsync(descriptor: int) -> None:
        nonlocal mutated
        if not mutated and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            mutated = True
            if mutation == "replace":
                os.replace(forged, output)
            else:
                output.chmod(0o644)
        real_fsync(descriptor)

    monkeypatch.setattr(evidence_module.os, "fsync", mutate_leaf_during_directory_fsync)

    with pytest.raises(ValueError, match="evidence write failed") as raised:
        gate0_module.write_appworld_gate0_summary(output, summary)
    assert raised.value.__cause__ is None
    assert mutated is True
    if mutation == "replace":
        assert output.read_bytes() == b"FORGED"
        assert stat.S_IMODE(output.stat().st_mode) == 0o644
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("schema_version", 2),
        ("pinned_appworld_commit", "0" * 40),
        ("pinned_appworld_package_version", "0.2.0"),
        ("pinned_appworld_data_version", "PRIVATE_DATA_CANARY"),
        ("pinned_appworld_base_db_version", "PRIVATE_DB_CANARY"),
        ("pinned_python_version", "3.11.14"),
        ("adapter_abi_sha256", "0" * 64),
        ("runner_abi_sha256", "0" * 64),
        ("screened_task_count", True),
        ("admitted_task_count", 1),
    ],
)
def test_mutated_gate0_summary_is_rejected_at_serialization(
    field_name: str,
    invalid_value: object,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    object.__setattr__(summary, field_name, invalid_value)

    with pytest.raises(ValueError, match="summary is invalid") as raised:
        summary.to_dict()
    assert "PRIVATE" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_gate0_summary_rejects_lexically_safe_unknown_diagnostic_code() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    empty = gate0_module.AppWorldTransformSummary(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="summary is invalid") as raised:
        gate0_module.AppWorldGate0Summary(
            mode="smoke",
            split="train",
            seed=100,
            workers=1,
            screened_task_count=0,
            excluded_task_count=0,
            admitted_task_count=0,
            clean=empty,
            l1=empty,
            l2=empty,
            diagnostic_counts=(("PRIVATE_TASK_ID_CANARY", 1),),
            execution_exception_count=0,
            cleanup_exception_count=0,
            gate_evaluable=False,
            formal_gate_passed=False,
            smoke_passed=False,
        )
    assert "PRIVATE_TASK_ID_CANARY" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("execution_count", "cleanup_count"),
    [(2, 0), (0, 2), (999, 777)],
)
def test_gate0_summary_rejects_unreachable_exception_counts(
    execution_count: int,
    cleanup_count: int,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    object.__setattr__(summary, "execution_exception_count", execution_count)
    object.__setattr__(summary, "cleanup_exception_count", cleanup_count)

    with pytest.raises(ValueError, match="summary is invalid"):
        summary.to_dict()


@pytest.mark.parametrize(
    "exception_field",
    ["execution_exception_count", "cleanup_exception_count"],
)
def test_gate0_summary_rejects_admission_with_any_lifecycle_exception(
    exception_field: str,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    empty = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    admitted = gate0_module.AppWorldTransformSummary(1, 1, 1, 1, 0)
    summary = replace(
        empty,
        screened_task_count=1,
        excluded_task_count=0,
        admitted_task_count=1,
        clean=admitted,
        l1=admitted,
        l2=admitted,
        diagnostic_counts=(),
        smoke_passed=True,
    )
    object.__setattr__(summary, exception_field, 1)
    object.__setattr__(summary, "smoke_passed", False)

    with pytest.raises(ValueError, match="summary is invalid"):
        summary.to_dict()


def test_gate0_summary_rejects_admission_diagnostic_without_attempt() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    object.__setattr__(summary, "diagnostic_counts", (("state.final_mismatch", 1),))

    with pytest.raises(ValueError, match="summary is invalid"):
        summary.to_dict()


def test_gate0_summary_rejects_unreachable_singleton_diagnostic_count() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    with pytest.raises(ValueError, match="summary is invalid"):
        gate0_module.AppWorldGate0Summary(
            mode="smoke",
            split="train",
            seed=100,
            workers=1,
            screened_task_count=1,
            excluded_task_count=1,
            admitted_task_count=0,
            clean=gate0_module.AppWorldTransformSummary(1, 1, 1, 0, 0),
            l1=gate0_module.AppWorldTransformSummary(1, 1, 0, 0, 0),
            l2=gate0_module.AppWorldTransformSummary(1, 1, 0, 0, 0),
            diagnostic_counts=(("state.final_mismatch", 999),),
            execution_exception_count=0,
            cleanup_exception_count=0,
            gate_evaluable=False,
            formal_gate_passed=False,
            smoke_passed=False,
        )


@pytest.mark.parametrize(
    "diagnostic_code",
    ["smoke.admission_exception", "smoke.invalid_admission_record"],
)
def test_gate0_summary_rejects_repeated_smoke_admission_diagnostic(
    diagnostic_code: str,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    with pytest.raises(ValueError, match="summary is invalid"):
        gate0_module.AppWorldGate0Summary(
            mode="smoke",
            split="train",
            seed=100,
            workers=1,
            screened_task_count=1,
            excluded_task_count=1,
            admitted_task_count=0,
            clean=gate0_module.AppWorldTransformSummary(1, 1, 1, 0, 0),
            l1=gate0_module.AppWorldTransformSummary(1, 1, 0, 0, 0),
            l2=gate0_module.AppWorldTransformSummary(1, 1, 0, 0, 0),
            diagnostic_counts=((diagnostic_code, 999),),
            execution_exception_count=0,
            cleanup_exception_count=0,
            gate_evaluable=False,
            formal_gate_passed=False,
            smoke_passed=False,
        )


@pytest.mark.parametrize(
    ("execution_count", "cleanup_count"),
    [(1, 0), (0, 1)],
)
def test_gate0_summary_rejects_contract_diagnostic_with_lifecycle_exception(
    execution_count: int,
    cleanup_count: int,
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    with pytest.raises(ValueError, match="summary is invalid"):
        gate0_module.AppWorldGate0Summary(
            mode="smoke",
            split="train",
            seed=100,
            workers=1,
            screened_task_count=1,
            excluded_task_count=1,
            admitted_task_count=0,
            clean=gate0_module.AppWorldTransformSummary(1, 1, 1, 0, 0),
            l1=gate0_module.AppWorldTransformSummary(1, 1, 0, 0, 0),
            l2=gate0_module.AppWorldTransformSummary(1, 1, 0, 0, 0),
            diagnostic_counts=(("state.final_mismatch", 1),),
            execution_exception_count=execution_count,
            cleanup_exception_count=cleanup_count,
            gate_evaluable=False,
            formal_gate_passed=False,
            smoke_passed=False,
        )


def test_gate0_summary_accepts_reset_hash_diagnostic_with_execution_exception() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    summary = gate0_module.AppWorldGate0Summary(
        mode="smoke",
        split="train",
        seed=100,
        workers=1,
        screened_task_count=1,
        excluded_task_count=1,
        admitted_task_count=0,
        clean=gate0_module.AppWorldTransformSummary(1, 1, 1, 0, 0),
        l1=gate0_module.AppWorldTransformSummary(1, 1, 0, 0, 0),
        l2=gate0_module.AppWorldTransformSummary(1, 1, 0, 0, 0),
        diagnostic_counts=(("pair.reset_hash_exception", 1),),
        execution_exception_count=1,
        cleanup_exception_count=0,
        gate_evaluable=False,
        formal_gate_passed=False,
        smoke_passed=False,
    )

    assert summary.diagnostic_counts == (("pair.reset_hash_exception", 1),)


@pytest.mark.parametrize(
    "diagnostic_code",
    [
        "trace.reference_effect_count",
        "trace.reference_effect_alignment",
        "trace.candidate_effect_count",
        "trace.candidate_effect_alignment",
    ],
)
def test_gate0_preserves_static_effect_diagnostic_codes(diagnostic_code: str) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    error = gate0_module._VariantAdmissionError(
        "paired AppWorld evidence failed",
        (diagnostic_code,),
    )

    assert error.diagnostic_codes == (diagnostic_code,)


def test_gate0_summary_rejects_repeated_exclusive_preflight_diagnostic() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    empty = gate0_module.AppWorldTransformSummary(0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="summary is invalid"):
        gate0_module.AppWorldGate0Summary(
            mode="smoke",
            split="train",
            seed=100,
            workers=1,
            screened_task_count=0,
            excluded_task_count=0,
            admitted_task_count=0,
            clean=empty,
            l1=empty,
            l2=empty,
            diagnostic_counts=(("smoke.preflight_failure", 2),),
            execution_exception_count=0,
            cleanup_exception_count=0,
            gate_evaluable=False,
            formal_gate_passed=False,
            smoke_passed=False,
        )


def test_gate0_summary_rejects_missing_screen_exclusion_diagnostic() -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    admitted = gate0_module.AppWorldTransformSummary(2, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="summary is invalid"):
        gate0_module.AppWorldGate0Summary(
            mode="smoke",
            split="train",
            seed=100,
            workers=1,
            screened_task_count=2,
            excluded_task_count=1,
            admitted_task_count=1,
            clean=admitted,
            l1=admitted,
            l2=admitted,
            diagnostic_counts=(),
            execution_exception_count=0,
            cleanup_exception_count=0,
            gate_evaluable=False,
            formal_gate_passed=False,
            smoke_passed=True,
        )


@pytest.mark.parametrize(
    ("clean", "l1", "l2", "diagnostics"),
    [
        pytest.param(
            (1, 1, 1, 1, 0),
            (1, 1, 1, 1, 0),
            (1, 1, 1, 1, 0),
            (("smoke.synthetic_failure", 1),),
            id="completed-families-without-admitted-task",
        ),
        pytest.param(
            (1, 1, 1, 0, 0),
            (1, 1, 1, 0, 0),
            (1, 1, 0, 0, 0),
            (("smoke.synthetic_failure", 1),),
            id="l1-attempted-before-clean-admission",
        ),
        pytest.param(
            (1, 1, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (("smoke.synthetic_failure", 1),),
            id="l2-eligible-without-selection-attempt",
        ),
        pytest.param(
            (1, 0, 0, 0, 1),
            (1, 1, 0, 0, 0),
            (1, 0, 0, 0, 1),
            (("smoke.synthetic_failure", 1),),
            id="l1-eligible-without-clean",
        ),
        pytest.param(
            (1, 0, 0, 0, 1),
            (1, 0, 0, 0, 1),
            (1, 0, 0, 0, 1),
            (),
            id="failed-summary-without-diagnostic",
        ),
    ],
)
def test_gate0_summary_rejects_impossible_cross_family_counters(
    clean: tuple[int, int, int, int, int],
    l1: tuple[int, int, int, int, int],
    l2: tuple[int, int, int, int, int],
    diagnostics: tuple[tuple[str, int], ...],
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    with pytest.raises(ValueError, match="summary is invalid"):
        gate0_module.AppWorldGate0Summary(
            mode="smoke",
            split="train",
            seed=100,
            workers=1,
            screened_task_count=1,
            excluded_task_count=1,
            admitted_task_count=0,
            clean=gate0_module.AppWorldTransformSummary(*clean),
            l1=gate0_module.AppWorldTransformSummary(*l1),
            l2=gate0_module.AppWorldTransformSummary(*l2),
            diagnostic_counts=diagnostics,
            execution_exception_count=0,
            cleanup_exception_count=0,
            gate_evaluable=False,
            formal_gate_passed=False,
            smoke_passed=False,
        )


def test_gate0_cli_writes_failed_summary_and_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import toolshift.benchmarks.appworld_gate0 as gate0_module

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "verify_appworld_gate0.py"
    spec = importlib.util.spec_from_file_location("verify_appworld_gate0", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    output = private_root / "result.json"
    summary = gate0_module.run_appworld_gate0_smoke(
        task_loader=lambda split: (),
        world_context_factory=pytest.fail,
        pin_checker=lambda: None,
    )
    run_calls = 0
    run_result = summary

    def run() -> object:
        nonlocal run_calls
        run_calls += 1
        return run_result

    monkeypatch.setattr(cli, "run_appworld_gate0_smoke", run)

    assert cli.main(["--output", str(output)]) == 1
    assert run_calls == 1
    assert json.loads(output.read_text(encoding="utf-8")) == summary.to_dict()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600

    admitted = gate0_module.AppWorldTransformSummary(1, 1, 1, 1, 0)
    run_result = replace(
        summary,
        screened_task_count=1,
        excluded_task_count=0,
        admitted_task_count=1,
        clean=admitted,
        l1=admitted,
        l2=admitted,
        diagnostic_counts=(),
        smoke_passed=True,
    )
    passed_output = private_root / "passed.json"
    assert cli.main(["--output", str(passed_output)]) == 0
    assert json.loads(passed_output.read_text(encoding="utf-8"))["smoke_passed"] is True
    assert run_calls == 2

    assert cli.main(["--output", "relative.json"]) == 2
    assert run_calls == 2

    for forbidden_option in ("--task-id", "--split", "--proxy", "--raw-log"):
        with pytest.raises(SystemExit) as raised:
            cli.main(["--output", str(output), forbidden_option, "PRIVATE_CANARY"])
        assert raised.value.code == 2
        captured = capsys.readouterr()
        assert "PRIVATE_CANARY" not in captured.out
        assert "PRIVATE_CANARY" not in captured.err
        assert str(output) not in captured.out
        assert str(output) not in captured.err
    assert run_calls == 2


def test_gate0_cli_writes_static_failed_summary_when_runner_preflight_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "verify_appworld_gate0.py"
    spec = importlib.util.spec_from_file_location("verify_appworld_gate0_preflight", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    monkeypatch.setenv("APPWORLD_ROOT", str(private_root))
    output = private_root / "preflight-failure.json"

    def fail_preflight() -> None:
        raise RuntimeError("PRIVATE_PREFLIGHT_CANARY")

    monkeypatch.setattr(cli, "run_appworld_gate0_smoke", fail_preflight)

    assert cli.main(["--output", str(output)]) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["smoke_passed"] is False
    assert payload["diagnostic_counts"] == {"smoke.preflight_failure": 1}
    assert "PRIVATE" not in output.read_text(encoding="utf-8")
