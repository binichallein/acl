"""Opt-in aggregate-only smoke for the pinned private AppWorld M3A runtime."""

from __future__ import annotations

import os

import pytest

from toolshift.benchmarks.appworld_gate0 import run_appworld_gate0_smoke

pytestmark = pytest.mark.appworld


def test_private_appworld_gate0_smoke() -> None:
    """Require one admitted clean/L1/L2 suite when explicitly enabled."""

    if os.environ.get("TOOLSHIFT_RUN_APPWORLD_GATE0_SMOKE") != "1":
        pytest.skip(
            "set TOOLSHIFT_RUN_APPWORLD_GATE0_SMOKE=1 for the private AppWorld Gate 0 smoke"
        )

    summary = run_appworld_gate0_smoke()

    assert summary.mode == "smoke"
    assert summary.split == "train"
    assert summary.seed == 100
    assert summary.workers == 1
    assert summary.screened_task_count >= 1
    assert summary.admitted_task_count == 1
    assert summary.clean.admitted_count == 1
    assert summary.l1.admitted_count == 1
    assert summary.l2.admitted_count == 1
    assert summary.diagnostic_counts == ()
    assert summary.execution_exception_count == 0
    assert summary.cleanup_exception_count == 0
    assert summary.gate_evaluable is False
    assert summary.formal_gate_passed is False
    assert summary.smoke_passed is True
