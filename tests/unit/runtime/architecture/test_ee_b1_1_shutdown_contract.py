# © Artur Czarnecki. All rights reserved.

"""EE-B1.1 — ExecutionRuntime graceful shutdown phase contract."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_reliability import (
    EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER,
    ExecutionRuntimeShutdownPhase,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b1_1_shutdown_phase_order_is_canonical() -> None:
    assert EXECUTION_RUNTIME_SHUTDOWN_PHASE_ORDER == (
        ExecutionRuntimeShutdownPhase.STOP_ACCEPTING_NEW_WORK,
        ExecutionRuntimeShutdownPhase.DRAIN_ACTIVE_EXECUTIONS,
        ExecutionRuntimeShutdownPhase.FLUSH_REQUIRED_EVIDENCE,
        ExecutionRuntimeShutdownPhase.PERSIST_FINAL_STATE,
        ExecutionRuntimeShutdownPhase.TERMINATE_WORKERS,
    )


def test_ee_b1_1_shutdown_contract_documented_in_reliability_model() -> None:
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    doc = (
        repo
        / "docs"
        / "project"
        / "maintainers"
        / "architecture"
        / "EXECUTION_ENGINE_RELIABILITY_MODEL.md"
    )
    text = doc.read_text(encoding="utf-8")
    assert "STOP_ACCEPTING_NEW_WORK" in text
    assert "TERMINATE_WORKERS" in text
    assert "reopening sealed attempts" in text.lower() or "sealed attempts" in text.lower()
