"""Synthetic tests for paired AppWorld Gate 0 contract admission."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, replace
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
                    "pass_count": 0,
                    "fail_count": 1,
                    "total_count": 1,
                    "num_tests": 1,
                    "success": False,
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
        ("pair.valid-code_0000", "PRIVATE INVALID CODE CANARY"),
    )

    assert repr(error) == "_VariantAdmissionError(<private>)"
    assert error.diagnostic_codes == (
        "pair.valid-code_0000",
        "pair.unsafe_diagnostic",
    )
    assert str(error) == "paired AppWorld execution failed"
    assert "PRIVATE" not in repr(error)
    assert "PRIVATE" not in str(error)
