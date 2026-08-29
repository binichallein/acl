"""Opt-in lifecycle smoke against the pinned external AppWorld checkout."""

from __future__ import annotations

import os

import pytest

from toolshift.benchmarks.appworld_replay import (
    FORMAL_SEED,
    ReplayMode,
    current_checkout_revision,
    load_appworld_task_set,
    require_pinned_appworld_revision,
    run_replay_verification,
)

pytestmark = pytest.mark.appworld


def test_pinned_appworld_fresh_world_oracle_lifecycle() -> None:
    """Create, freshly reset, execute native oracle APIs, evaluate, and close."""

    if os.environ.get("TOOLSHIFT_RUN_APPWORLD_SMOKE") != "1":
        pytest.skip("set TOOLSHIFT_RUN_APPWORLD_SMOKE=1 for the pinned AppWorld smoke")

    require_pinned_appworld_revision(current_checkout_revision())
    task_set = load_appworld_task_set(smoke=True)
    summary = run_replay_verification(
        task_set,
        mode=ReplayMode.SMOKE,
        seed=FORMAL_SEED,
        repetitions=2,
        workers=1,
    )

    assert summary.task_count == 1
    assert summary.episode_count == 2
    assert summary.initial_state_match_rate == 1.0
    assert summary.trace_match_rate == 1.0
    assert summary.task_consistency_rate == 1.0
    assert summary.oracle_success_rate == 1.0
    assert summary.execution_failure_count == 0
    assert summary.exception_count == 0
    assert summary.smoke_passed is True
    assert summary.gate_evaluable is False
    assert summary.gate_passed is False
