# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-D / F-D4-13: all six lifecycle metrics reach the export bridge; export is opt-in."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from intergrax.hosting import (
    HostedApplicationComponentRegistration,
    HostedApplicationEventType,
    HostedApplicationLifecycleState,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
    ObservabilityHostedApplicationEventPublisher,
    RestartPolicy,
)
from intergrax.hosting.contracts.policies import RestartMode
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.metrics import HostedApplicationLifecycleMetric, lifecycle_metric_event
from intergrax.runtime.observability.export_boundary import InMemoryObservabilityExporter
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from tests.qualification.pdc_x.d4._support import (
    APP_ID,
    JournalRuntime,
    OrderedComponent,
    SystemClock,
    build_engine,
    definition_of,
    make_profile,
)

pytestmark = [pytest.mark.integration]

_WAIT_SECONDS = 30.0
_ALL_SIX = {metric.value for metric in HostedApplicationLifecycleMetric}


def _metric_names(exporter: InMemoryObservabilityExporter) -> set[str]:
    return {
        str(envelope.sanitized_application_attributes.attributes["hosting.metric_name"])
        for envelope in exporter.envelopes
        if envelope.event_type == HostedApplicationEventType.LIFECYCLE_METRIC.value
        and envelope.sanitized_application_attributes is not None
    }


def _metric_event(metric: HostedApplicationLifecycleMetric):
    labelled = {HostedApplicationLifecycleMetric.DEGRADED, HostedApplicationLifecycleMetric.EXIT_CLASS}
    return lifecycle_metric_event(
        metric,
        application_id=APP_ID,
        instance_id="instance-001",
        lifecycle_state=HostedApplicationLifecycleState.READY,
        value=1.0,
        label="label" if metric in labelled else "",
        occurred_at=SystemClock().now(),
    )


async def test_all_six_metrics_survive_the_export_bridge_when_enabled() -> None:
    exporter = InMemoryObservabilityExporter()
    bridge = ObservabilityHostedApplicationEventPublisher(exporter, policy=ObservabilityExportPolicy(enabled=True))
    for metric in HostedApplicationLifecycleMetric:
        await bridge.publish(_metric_event(metric))
    assert len(_ALL_SIX) == 6
    assert _metric_names(exporter) == _ALL_SIX


async def test_the_default_publisher_exports_nothing() -> None:
    """Pins the documented opt-in: the default export policy is disabled."""
    exporter = InMemoryObservabilityExporter()
    bridge = ObservabilityHostedApplicationEventPublisher(exporter)
    for metric in HostedApplicationLifecycleMetric:
        await bridge.publish(_metric_event(metric))
    assert exporter.envelopes == []


async def test_a_supervised_restart_cycle_exports_every_metric(tmp_path: Path) -> None:
    exporter = InMemoryObservabilityExporter()
    bridge = ObservabilityHostedApplicationEventPublisher(exporter, policy=ObservabilityExportPolicy(enabled=True))
    journal: list[str] = []
    profile = make_profile(
        runtime=JournalRuntime(journal),
        components=(
            HostedApplicationComponentRegistration(
                component=OrderedComponent("degraded", journal, healthy=False),
                required=False,
                start_timeout_seconds=5.0,
                stop_timeout_seconds=5.0,
            ),
        ),
        restart=RestartPolicy(
            mode=RestartMode.ON_FAILURE,
            max_attempts=3,
            initial_backoff_seconds=0.05,
            max_backoff_seconds=0.2,
            jitter_ratio=0.0,
        ),
    )
    definition = definition_of(profile)
    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    counter = iter(range(100))

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        engine, _, _ = build_engine(
            launch.definition,
            tmp_path,
            publisher=bridge,  # type: ignore[arg-type]
            control=launch.control,
            instance_id=launch.instance_id,
        )
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=bridge,
        clock=clock,
        instance_id_generator=lambda: f"export-attempt-{next(counter)}",
    )

    async def _wait_for_metric(name: str) -> None:
        deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
        while name not in _metric_names(exporter):
            assert asyncio.get_running_loop().time() < deadline, f"metric never exported: {name}"
            await asyncio.sleep(0.05)

    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_for_metric("ready")
        control.request_restart("test.export_restart")
        await _wait_for_metric("restart_count")
        control.request_shutdown("test.done")
        await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("test.abort")
            await asyncio.gather(run, return_exceptions=True)
    assert _metric_names(exporter) == _ALL_SIX
