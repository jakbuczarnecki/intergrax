# © Artur Czarnecki. All rights reserved.

"""PDC-D4-R005 (xd_05): shutdown drains in-flight background work.

The in-flight work is registered through the platform background-task contract
(``TaskRegistry`` / ``TaskDefinition``) and tracked through the hosting active-work
port, so the real engine shutdown executor drains it. PDC-P6-01 re-points this case
at the integrated baseline.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from intergrax.background_tasks.definition import TaskDefinition
from intergrax.background_tasks.registry import TaskRegistry
from intergrax.hosting import (
    HostedApplicationEventType,
    HostedApplicationExitKind,
    HostedApplicationLifecycleState,
    ShutdownPolicy,
)
from intergrax.hosting.shutdown import (
    HostedApplicationShutdownPhase,
    HostedApplicationShutdownPhaseOutcome,
)
from intergrax.hosting.supervisor.classification import HostedApplicationExitClassifier
from pydantic import BaseModel
from tests.qualification.pdc_x.d4._support import (
    APP_ID,
    AsyncioActiveWork,
    JournalRuntime,
    SystemClock,
    build_engine,
    definition_of,
    make_profile,
)

pytestmark = [pytest.mark.integration]

TASK_NAME = "pdc.xd05.background_work"


class _WorkPayload(BaseModel):
    seconds: float


def _blocking_handler(payload: _WorkPayload) -> str:
    time.sleep(payload.seconds)
    return "done"


def _registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register(
        TaskDefinition(
            task_name=TASK_NAME,
            payload_schema=_WorkPayload,
            handler=_blocking_handler,
        )
    )
    return registry


def _registered_work(registry: TaskRegistry, seconds: float):
    definition = registry.resolve(TASK_NAME)

    async def work() -> None:
        await asyncio.to_thread(definition.handler, _WorkPayload(seconds=seconds))

    return work


async def test_xd05_shutdown_drains_in_flight_background_work(tmp_path: Path) -> None:
    registry = _registry()
    controller = AsyncioActiveWork()
    profile = make_profile(
        runtime=JournalRuntime(),
        shutdown=ShutdownPolicy.drain_then_cancel(
            drain_timeout_seconds=10.0,
            cancel_timeout_seconds=2.0,
            flush_timeout_seconds=1.0,
        ),
    )
    engine, publisher, _ = build_engine(definition_of(profile), tmp_path, active_work_controller=controller)
    await engine.start()
    assert engine.accepts_new_work

    controller.submit("job-1", _registered_work(registry, 0.4))
    controller.submit("job-2", _registered_work(registry, 0.2))
    assert controller.active_work_count() == 2

    result = await asyncio.wait_for(engine.stop(reason_code="test.drain"), timeout=20.0)

    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED
    # In-flight work finished (drained), it was not cancelled.
    assert sorted(controller.completed) == ["job-1", "job-2"]
    assert controller.cancelled == []
    assert controller.active_work_count() == 0

    execution = result.diagnostics.shutdown_execution
    assert execution is not None
    assert execution.active_work_before == 2
    assert execution.active_work_after == 0
    assert execution.timed_out is False
    assert execution.forced is False
    phases = {record.phase: record for record in execution.phase_records}
    assert phases[HostedApplicationShutdownPhase.STOP_INTAKE].outcome is (
        HostedApplicationShutdownPhaseOutcome.COMPLETED
    )
    assert phases[HostedApplicationShutdownPhase.DRAIN].outcome is (HostedApplicationShutdownPhaseOutcome.COMPLETED)
    assert HostedApplicationShutdownPhase.CANCEL not in phases

    # Intake is closed and the host no longer accepts new work once shutdown began.
    with pytest.raises(RuntimeError):
        controller.submit("late-job", _registered_work(registry, 0.0))
    assert engine.accepts_new_work is False

    stopping = publisher.of_type(HostedApplicationEventType.APPLICATION_STOPPING)
    stopped = publisher.of_type(HostedApplicationEventType.APPLICATION_STOPPED)
    assert len(stopping) == 1 and len(stopped) == 1

    exit_record = HostedApplicationExitClassifier().classify_terminal_result(
        result,
        application_id=APP_ID,
        instance_id="instance-001",
        profile_digest=engine.definition.profile_digest,
        occurred_at=SystemClock().now(),
        shutdown_execution=execution,
    )
    assert exit_record.exit_kind is HostedApplicationExitKind.CLEAN_STOP


async def test_xd05_work_that_outlives_the_drain_budget_is_cancelled_and_reported(
    tmp_path: Path,
) -> None:
    controller = AsyncioActiveWork()
    profile = make_profile(
        runtime=JournalRuntime(),
        shutdown=ShutdownPolicy.drain_then_cancel(
            drain_timeout_seconds=0.3,
            cancel_timeout_seconds=2.0,
            flush_timeout_seconds=1.0,
        ),
    )
    engine, _, _ = build_engine(definition_of(profile), tmp_path, active_work_controller=controller)
    await engine.start()

    async def stuck() -> None:
        await asyncio.sleep(30.0)

    controller.submit("stuck-job", stuck)

    result = await asyncio.wait_for(engine.stop(reason_code="test.drain.budget"), timeout=20.0)

    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED
    assert controller.cancelled == ["stuck-job"]
    assert controller.active_work_count() == 0
    execution = result.diagnostics.shutdown_execution
    assert execution is not None
    assert execution.forced is True
    phases = {record.phase: record for record in execution.phase_records}
    assert phases[HostedApplicationShutdownPhase.DRAIN].outcome is (HostedApplicationShutdownPhaseOutcome.TIMED_OUT)
    assert HostedApplicationShutdownPhase.CANCEL in phases

    exit_record = HostedApplicationExitClassifier().classify_terminal_result(
        result,
        application_id=APP_ID,
        instance_id="instance-001",
        profile_digest=engine.definition.profile_digest,
        occurred_at=SystemClock().now(),
        shutdown_execution=execution,
    )
    assert exit_record.exit_kind is HostedApplicationExitKind.FORCED_TERMINATION
