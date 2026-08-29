"""Shared fixtures for behavioral-equivalence contract tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from toolshift.adapters.semantic import SemanticAdapter
from toolshift.contracts.denotation import DenotationCase
from toolshift.contracts.schema import (
    ContractSuiteResult,
    SchemaProbe,
    evaluate_contract_suite,
)
from toolshift.contracts.state import StateCase, StateEvidence
from toolshift.contracts.trace import (
    PhysicalCallEffect,
    TraceCase,
    TraceEvidence,
)
from toolshift.types import (
    ExecutionTrace,
    JSONValue,
    SchemaVariant,
    SemanticAction,
    SurfaceToolSpec,
    canonical_json_bytes,
)


def _tool(name: str = "search") -> SurfaceToolSpec:
    return SurfaceToolSpec(
        name=name,
        description="Search documents",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


def _variant(*names: str) -> SchemaVariant:
    return SchemaVariant(
        variant_id="identity-v1",
        tools=[_tool(name) for name in (names or ("search",))],
        manifest={"operator": "identity", "seed": 7},
    )


def _surface_call(name: str = "search", value: JSONValue = "alpha") -> dict[str, JSONValue]:
    return {"name": name, "arguments": {"value": value}}


class _IdentityAdapter(SemanticAdapter):
    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        name = surface_call.get("name")
        arguments = surface_call.get("arguments")
        if not isinstance(name, str) or name not in {tool.name for tool in self.variant.tools}:
            raise ValueError("unknown surface tool")
        if not isinstance(arguments, Mapping):
            raise ValueError("surface call arguments must be a mapping")
        return (SemanticAction(name, arguments),)

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        return ({"name": action.name, "arguments": action.arguments},)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        if len(actions) == 0 and len(base_observation_groups) == 0:
            return {"status": "noop"}
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        return trace.semantic_actions


class _BadSchemaAdapter(_IdentityAdapter):
    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        actions = super().surface_to_semantic(surface_call)
        return (SemanticAction("wrong_action", actions[0].arguments),)


class _NondeterministicAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._calls = 0

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self._calls += 1
        actions = super().surface_to_semantic(surface_call)
        if self._calls % 2 == 0:
            return (SemanticAction(actions[0].name, {"value": "changed"}),)
        return actions


class _ExplodingAdapter(_IdentityAdapter):
    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        raise RuntimeError("payload at /private/path: {'secret': 17}")


class _EmptyCompileAdapter(_IdentityAdapter):
    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        return ()


class _NondeterministicCompileAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._compile_calls = 0

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        self._compile_calls += 1
        suffix: JSONValue = self._compile_calls
        return ({"name": action.name, "arguments": {"value": suffix}},)


class _TwoCallAdapter(_IdentityAdapter):
    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        return (
            {"name": action.name, "arguments": action.arguments},
            {"name": "audit", "arguments": {}},
        )


class _BadObservationAdapter(_IdentityAdapter):
    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        return {"ok": 1}


class _NoOpAdapter(_IdentityAdapter):
    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        return ()


class _ExplodingDenotationAdapter(_IdentityAdapter):
    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        raise RuntimeError("base payload /private/call {'datum': 91}")

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        raise RuntimeError("observation payload /private/observation 92")


class _ActualPipelineAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self.parsed_actions: list[SemanticAction] = []
        self.parsed_runs: list[tuple[SemanticAction, ...]] = []
        self.compiled_actions: list[SemanticAction] = []
        self.wrapped_actions: list[tuple[SemanticAction, ...]] = []
        self.events: list[str] = []

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        action = SemanticAction("search", {"value": "alpha"})
        self.parsed_actions.append(action)
        parsed_run = (action,)
        self.parsed_runs.append(parsed_run)
        self.events.append("parse")
        return parsed_run

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        self.compiled_actions.append(action)
        self.events.append("compile")
        value = "changed" if action is self.parsed_actions[1] else "alpha"
        return (_surface_call(value=value),)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self.wrapped_actions.append(actions)
        self.events.append("wrap")
        if actions and actions[0] is self.parsed_actions[1]:
            return {"items": [2]}
        return {"items": [1]}


class _OneParseFailureAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self.parse_calls = 0
        self.compiled_actions: list[SemanticAction] = []
        self.wrapped_actions: list[tuple[SemanticAction, ...]] = []

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self.parse_calls += 1
        if self.parse_calls == 1:
            raise RuntimeError("parser payload /private/parse 93")
        return (SemanticAction("search", {"value": "alpha"}),)

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        self.compiled_actions.append(action)
        return (_surface_call(),)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self.wrapped_actions.append(actions)
        return {"items": [1]}


class _SecondCompileDriftAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self.compile_counts: dict[int, int] = {}

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        action_id = id(action)
        count = self.compile_counts.get(action_id, 0) + 1
        self.compile_counts[action_id] = count
        value = "changed" if count == 2 else "alpha"
        return (_surface_call(value=value),)


class _StageRebindingAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant, stage: str) -> None:
        super().__init__(variant)
        self._stage = stage
        self._rebound_variant = SchemaVariant(
            "rebound-v1",
            variant.tools,
            {"operator": "rebound", "seed": 8},
        )

    def _maybe_rebind(self, stage: str) -> None:
        if self._stage == stage:
            self._variant = self._rebound_variant

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self._maybe_rebind("parse")
        name = surface_call["name"]
        arguments = surface_call["arguments"]
        assert isinstance(name, str)
        assert isinstance(arguments, Mapping)
        return (SemanticAction(name, arguments),)

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        self._maybe_rebind("compile")
        return ({"name": action.name, "arguments": action.arguments},)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self._maybe_rebind("wrap")
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self._maybe_rebind("canonicalize")
        return trace.semantic_actions


class _InPlaceVariantMutationAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant) -> None:
        super().__init__(variant)
        self._mutated = False

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        if not self._mutated:
            replacement = SchemaVariant(
                self._variant.variant_id,
                self._variant.tools,
                {"operator": "mutated", "seed": 9},
            )
            object.__setattr__(self._variant, "manifest", replacement.manifest)
            object.__setattr__(
                self._variant,
                "_manifest_canonical",
                replacement._manifest_canonical,
            )
            self._mutated = True
        return super().surface_to_semantic(surface_call)


class _TransientStageRebindingAdapter(_IdentityAdapter):
    def __init__(self, variant: SchemaVariant, stage: str) -> None:
        super().__init__(variant)
        self._original_variant = variant
        self._rebound_variant = SchemaVariant(
            "transient-v1",
            variant.tools,
            {"operator": "transient", "seed": 10},
        )
        self._stage = stage

    def _toggle_binding(self, stage: str) -> None:
        if self._stage == stage:
            self._variant = (
                self._rebound_variant
                if self._variant is self._original_variant
                else self._original_variant
            )

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self._toggle_binding("parse")
        name = surface_call["name"]
        arguments = surface_call["arguments"]
        assert isinstance(name, str)
        assert isinstance(arguments, Mapping)
        return (SemanticAction(name, arguments),)

    def semantic_to_base_calls(self, action: SemanticAction) -> tuple[Mapping[str, JSONValue], ...]:
        self._toggle_binding("compile")
        return ({"name": action.name, "arguments": action.arguments},)

    def base_observation_to_surface(
        self,
        surface_call: Mapping[str, JSONValue],
        actions: tuple[SemanticAction, ...],
        base_observation_groups: tuple[tuple[JSONValue, ...], ...],
    ) -> JSONValue:
        self._toggle_binding("wrap")
        return base_observation_groups[0][0]

    def canonicalize_trace(self, trace: ExecutionTrace) -> tuple[SemanticAction, ...]:
        self._toggle_binding("canonicalize")
        return trace.semantic_actions


class _DenotationListSwapAdapter(_IdentityAdapter):
    def __init__(
        self,
        variant: SchemaVariant,
        mutable_steps: list[DenotationCase],
        replacement: DenotationCase,
    ) -> None:
        super().__init__(variant)
        self._mutable_steps = mutable_steps
        self._replacement = replacement
        self._parse_calls = 0

    def surface_to_semantic(
        self, surface_call: Mapping[str, JSONValue]
    ) -> tuple[SemanticAction, ...]:
        self._parse_calls += 1
        if self._parse_calls == 3:
            self._mutable_steps[0] = self._replacement
        return super().surface_to_semantic(surface_call)


class _AlwaysEqualDigest(str):
    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


def _replace_denotation_case_content(
    target: DenotationCase,
    replacement: DenotationCase,
) -> None:
    for name in (
        "case_id",
        "surface_call",
        "expected_actions",
        "expected_base_call_groups",
        "base_observation_groups",
        "expected_surface_observation",
        "_snapshot_fingerprint",
    ):
        object.__setattr__(target, name, getattr(replacement, name))


def _replace_state_case_content(target: StateCase, replacement: StateCase) -> None:
    for name in ("case_id", "episode_id", "_snapshot_fingerprint"):
        object.__setattr__(target, name, getattr(replacement, name))


def _replace_trace_case_content(target: TraceCase, replacement: TraceCase) -> None:
    for name in ("case_id", "episode_id", "_snapshot_fingerprint"):
        object.__setattr__(target, name, getattr(replacement, name))


def _replace_variant_content(
    target: SchemaVariant,
    replacement: SchemaVariant,
) -> None:
    for name in (
        "variant_id",
        "tools",
        "manifest",
        "_manifest_canonical",
    ):
        object.__setattr__(target, name, getattr(replacement, name))


class _LayerBoundaryVariantAdapter(_IdentityAdapter):
    def __init__(
        self,
        original: SchemaVariant,
        intermediate: SchemaVariant,
    ) -> None:
        super().__init__(original)
        self._original_variant = original
        self._intermediate_variant = intermediate
        self._serve_intermediate = False
        self._intermediate_reads = 0

    def begin_intermediate_window(self) -> None:
        self._serve_intermediate = True
        self._intermediate_reads = 0

    @property
    def variant(self) -> SchemaVariant:
        if self._serve_intermediate:
            self._intermediate_reads += 1
            if self._intermediate_reads <= 3:
                return self._intermediate_variant
        return self._original_variant


def _state_evidence(**overrides: str) -> StateEvidence:
    values = {
        "reference_initial_sha256": "a" * 64,
        "candidate_initial_sha256": "a" * 64,
        "reference_final_sha256": "b" * 64,
        "candidate_final_sha256": "b" * 64,
        "reference_post_reset_sha256": "a" * 64,
        "candidate_post_reset_sha256": "a" * 64,
        "reference_collateral_digest": "collateral-clean",
        "candidate_collateral_digest": "collateral-clean",
    }
    values.update(overrides)
    return StateEvidence(**values)


def _effect(
    call: Mapping[str, JSONValue], index: int = 0, digest: str = "effect-clean"
) -> PhysicalCallEffect:
    return PhysicalCallEffect(
        index,
        hashlib.sha256(canonical_json_bytes(call)).hexdigest(),
        digest,
    )


def _single_step_trace_fixture() -> tuple[
    DenotationCase, TraceCase, ExecutionTrace, PhysicalCallEffect
]:
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    step = DenotationCase(
        "denotation-trace-step",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    trace = ExecutionTrace([call], [action], [call])
    return step, TraceCase("trace-case", "episode-trace"), trace, _effect(call)


def _identity_suite_components(
    *,
    state_episode: str = "episode-suite",
    trace_episode: str = "episode-suite",
) -> tuple[
    SchemaVariant,
    _IdentityAdapter,
    SchemaProbe,
    DenotationCase,
    StateCase,
    TraceCase,
    TraceEvidence,
]:
    variant = _variant()
    adapter = _IdentityAdapter(variant)
    action = SemanticAction("search", {"value": "alpha"})
    call = _surface_call()
    probe = SchemaProbe("schema-suite-helper", "search", call, [action])
    step = DenotationCase(
        "denotation-suite-helper",
        call,
        [action],
        [[call]],
        [[{"items": [1]}]],
        {"items": [1]},
    )
    state_case = StateCase("state-suite-helper", state_episode)
    trace_case = TraceCase("trace-suite-helper", trace_episode)
    trace = ExecutionTrace([call], [action], [call])
    effect = _effect(call)
    evidence = TraceEvidence(trace, trace, [step], 1, 1, [effect], [effect])
    return variant, adapter, probe, step, state_case, trace_case, evidence


def _evaluate_identity_components(
    *,
    state_episode: str = "episode-suite",
    trace_episode: str = "episode-suite",
    adapter_override: SemanticAdapter | None = None,
    state_provider_override: object | None = None,
    trace_evidence_override: TraceEvidence | None = None,
) -> tuple[SchemaVariant, ContractSuiteResult]:
    variant, adapter, probe, step, state_case, trace_case, trace_evidence = (
        _identity_suite_components(
            state_episode=state_episode,
            trace_episode=trace_episode,
        )
    )
    selected_adapter = adapter if adapter_override is None else adapter_override
    selected_state_provider = (
        (lambda _: _state_evidence())
        if state_provider_override is None
        else state_provider_override
    )
    selected_trace_evidence = (
        trace_evidence if trace_evidence_override is None else trace_evidence_override
    )
    suite = evaluate_contract_suite(
        variant,
        selected_adapter,
        [probe],
        [step],
        [state_case],
        selected_state_provider,
        [trace_case],
        lambda _: selected_trace_evidence,
    )
    return variant, suite
