# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-D / F-D4-12: event and metric publishes cannot hang shutdown or supervision."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from intergrax.hosting import (
    HostedApplicationEngine,
    HostedApplicationEvent,
    HostedApplicationEventType,
    HostedApplicationLifecycleState,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
    LifecyclePolicy,
)
from intergrax.hosting.control import HostedApplicationControlCoordinator
from tests.qualification.pdc_x.d4._support import (
    JournalRuntime,
    RecordingPublisher,
    SystemClock,
    build_engine,
    definition_of,
    make_profile,
)

pytestmark = [pytest.mark.integration]

_BOUND_SECONDS = 0.3
_WAIT_SECONDS = 30.0


@dataclass
class _HangingPublisher:
    """Publisher that hangs once armed and records every call."""

    armed: bool = False
    calls: list[HostedApplicationEvent] = field(default_factory=list)

    async def publish(self, event: HostedApplicationEvent) -> None:
        self.calls.append(event)
        if self.armed:
            await asyncio.Event().wait()


def _profile():
    return make_profile(
        runtime=JournalRuntime(),
        lifecycle=LifecyclePolicy(default_observer_hook_timeout_seconds=_BOUND_SECONDS),
    )


async def test_hanging_downstream_publisher_does_not_block_engine_stop(tmp_path: Path) -> None:
    publisher = _HangingPublisher()
    engine, _, _ = build_engine(definition_of(_profile()), tmp_path, publisher=publisher)  # type: ignore[arg-type]
    await engine.start()
    publisher.armed = True

    result = await asyncio.wait_for(engine.stop(reason_code="test.hanging_publisher"), _WAIT_SECONDS)

    assert result.terminal_state is HostedApplicationLifecycleState.STOPPED
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.instance_lease_released
    assert "event_publish_timed_out" in {record.reason_code for record in diagnostics.secondary_failures}


async def test_hanging_publisher_does_not_block_the_supervisor_failure_path(tmp_path: Path) -> None:
    publisher = _HangingPublisher(armed=True)
    definition = definition_of(_profile())
    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext) -> HostedApplicationEngine:
        raise RuntimeError("engine factory failed")

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,  # type: ignore[arg-type]
        clock=clock,
        instance_id_generator=lambda: "hanging-attempt-0",
    )
    result = await asyncio.wait_for(supervisor.run(), _WAIT_SECONDS)
    assert result.final_exit.exit_kind.value == "supervisor_error"
    assert len(publisher.calls) >= 2, "the metric and the failure event were both attempted"


async def test_prompt_publisher_still_receives_every_event_in_order(tmp_path: Path) -> None:
    """Negative control: bounding must not drop or reorder events of a healthy publisher."""
    publisher = RecordingPublisher()
    engine, _, _ = build_engine(definition_of(_profile()), tmp_path, publisher=publisher)
    await engine.start()
    await engine.stop(reason_code="test.prompt")

    types = [event.event_type for event in publisher.events]
    assert types[0] is HostedApplicationEventType.INSTANCE_ACQUIRED
    assert types[-1] is HostedApplicationEventType.APPLICATION_STOPPED
    metrics = {str(event.payload["metric"]) for event in publisher.of_type(HostedApplicationEventType.LIFECYCLE_METRIC)}
    assert {"start", "ready", "drain_duration", "exit_class"} <= metrics
    assert not engine.diagnostics_snapshot().secondary_failures
