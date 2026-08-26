# © Artur Czarnecki. All rights reserved.

"""Unit tests for terminal execution diagnostic trigger and bridge (ONE-SPINE-3)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge import (
    invoke_terminal_execution_diagnostics,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_trigger import (
    TerminalExecutionDiagnosticTrigger,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from tests.unit.runtime.diagnostics.test_diagnostic_orchestrator import (
    _build_orchestrator,
    _seed_retry_violation_sequence,
)

pytestmark = pytest.mark.unit

_OBSERVED_AT = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def test_trigger_builds_single_execution_orchestration_request() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    trigger = TerminalExecutionDiagnosticTrigger(orchestrator)

    result = trigger.trigger_for_terminal_execution(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        observed_at=_OBSERVED_AT,
    )

    assert len(result.execution_results) == 1
    assert result.execution_results[0].task_id == task_id
    assert result.execution_results[0].run_id == run_id


def test_trigger_replay_is_idempotent_for_same_execution() -> None:
    orchestrator, runtime_store, _, persistence = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    trigger = TerminalExecutionDiagnosticTrigger(orchestrator)

    first = trigger.trigger_for_terminal_execution(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        observed_at=_OBSERVED_AT,
    )
    second = trigger.trigger_for_terminal_execution(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        observed_at=_OBSERVED_AT,
    )

    assert first.execution_results[0].assessment.has_findings
    assert first.lifecycle_result.created == ()
    assert second.lifecycle_result.created == ()
    assert second.lifecycle_result.updated == ()
    assert second.lifecycle_result.unchanged == ()
    assert persistence.list_for_tenant("tenant-a") == ()


def test_bridge_returns_none_when_trigger_not_configured() -> None:
    result = invoke_terminal_execution_diagnostics(
        None,
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        observed_at=_OBSERVED_AT,
    )
    assert result is None


def test_bridge_surfaces_orchestrator_failure_without_raising() -> None:
    failing = MagicMock(spec=TerminalExecutionDiagnosticTrigger)
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    result = invoke_terminal_execution_diagnostics(
        failing,
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        observed_at=_OBSERVED_AT,
    )

    assert result is None


def test_trigger_requires_timezone_aware_observed_at() -> None:
    orchestrator, runtime_store, _, _ = _build_orchestrator()
    task_id, run_id = _seed_retry_violation_sequence(runtime_store)
    trigger = TerminalExecutionDiagnosticTrigger(orchestrator)

    with pytest.raises(ValueError, match="timezone-aware"):
        trigger.trigger_for_terminal_execution(
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            observed_at=datetime(2026, 8, 26, 12, 0),
        )


def test_clean_execution_sequence_produces_no_problem() -> None:
    orchestrator, runtime_store, _, persistence = _build_orchestrator()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    for event_type in (
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.TASK_COMPLETED,
    ):
        runtime_store.append(
            sample_runtime_event(
                tenant_id="tenant-a",
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ).model_copy(update={"event_type": event_type}),
            tenant_id="tenant-a",
        )
    trigger = TerminalExecutionDiagnosticTrigger(orchestrator)

    result = trigger.trigger_for_terminal_execution(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        observed_at=_OBSERVED_AT,
    )

    assert result.lifecycle_result.created == ()
    assert persistence.list_for_tenant("tenant-a") == ()
