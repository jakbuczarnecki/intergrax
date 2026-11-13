# © Artur Czarnecki. All rights reserved.

"""PDC-D4-R009 (APP-HOST-3D): lifecycle metrics through the canonical observability port.

Start, ready, degraded, drain duration, restart count and exit class are emitted as typed
hosting events on the existing event publisher port, and projected by the existing
observability export bridge. Hosting owns no metrics store or backend.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import intergrax.hosting.metrics as hosting_metrics
import pytest
from intergrax.hosting import (
    HostedApplicationComponentRegistration,
    HostedApplicationEventType,
    HostedApplicationSupervisor,
    HostedApplicationSupervisorLaunchContext,
    ObservabilityHostedApplicationEventPublisher,
    RestartPolicy,
    ShutdownPolicy,
)
from intergrax.hosting.contracts.policies import RestartMode
from intergrax.hosting.control import HostedApplicationControlCoordinator
from intergrax.hosting.metrics import HostedApplicationLifecycleMetric
from intergrax.runtime.observability.export_boundary import InMemoryObservabilityExporter
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy
from tests.qualification.pdc_x.d4._support import (
    AsyncioActiveWork,
    JournalRuntime,
    OrderedComponent,
    RecordingPublisher,
    SystemClock,
    build_engine,
    definition_of,
    make_profile,
)

pytestmark = [pytest.mark.integration]

_WAIT_SECONDS = 20.0


def _metrics(publisher: RecordingPublisher, metric: HostedApplicationLifecycleMetric) -> list[dict]:
    return [
        dict(event.payload)
        for event in publisher.of_type(HostedApplicationEventType.LIFECYCLE_METRIC)
        if event.payload["metric"] == metric.value
    ]


async def _wait_until(predicate, description: str) -> None:
    deadline = asyncio.get_running_loop().time() + _WAIT_SECONDS
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for: {description}")
        await asyncio.sleep(0.02)


async def test_r009_engine_emits_start_ready_degraded_and_drain_metrics(tmp_path: Path) -> None:
    journal: list[str] = []
    controller = AsyncioActiveWork()
    profile = make_profile(
        runtime=JournalRuntime(journal),
        components=(
            HostedApplicationComponentRegistration(
                component=OrderedComponent("optional_cache", journal, fail_start=True)
            ),
        ),
        shutdown=ShutdownPolicy.drain_then_cancel(
            drain_timeout_seconds=10.0, cancel_timeout_seconds=2.0, flush_timeout_seconds=1.0
        ),
    )
    engine, publisher, _ = build_engine(definition_of(profile), tmp_path, active_work_controller=controller)
    await engine.start()

    async def in_flight() -> None:
        await asyncio.sleep(0.3)

    controller.submit("drain-me", in_flight)
    await engine.stop(reason_code="test.metrics")

    (start,) = _metrics(publisher, HostedApplicationLifecycleMetric.START)
    assert start["value"] == 1 and start["unit"] == "count"

    (ready,) = _metrics(publisher, HostedApplicationLifecycleMetric.READY)
    assert ready["unit"] == "seconds" and ready["value"] >= 0.0

    degraded = _metrics(publisher, HostedApplicationLifecycleMetric.DEGRADED)
    assert degraded and degraded[0]["value"] == 1 and degraded[0]["label"] == "optional_cache"

    (drain,) = _metrics(publisher, HostedApplicationLifecycleMetric.DRAIN_DURATION)
    assert drain["unit"] == "seconds"
    assert 0.25 <= drain["value"] < 5.0

    # Ordering: the start metric follows the starting event; the drain metric follows stopping.
    order = [(event.event_type, event.payload.get("metric")) for event in publisher.events]
    starting = order.index((HostedApplicationEventType.APPLICATION_STARTING, None))
    start_metric = order.index((HostedApplicationEventType.LIFECYCLE_METRIC, "start"))
    stopping = order.index((HostedApplicationEventType.APPLICATION_STOPPING, None))
    drain_metric = order.index((HostedApplicationEventType.LIFECYCLE_METRIC, "drain_duration"))
    assert starting < start_metric < stopping < drain_metric


async def test_r009_supervisor_emits_restart_count_and_exit_class(tmp_path: Path) -> None:
    profile = make_profile(
        runtime=JournalRuntime(),
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
    publisher = RecordingPublisher()
    counter = iter(range(100))

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        engine, _, _ = build_engine(
            launch.definition,
            tmp_path,
            publisher=publisher,
            control=launch.control,
            instance_id=launch.instance_id,
        )
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: f"metrics-attempt-{next(counter)}",
    )
    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_until(
            lambda: len(_metrics(publisher, HostedApplicationLifecycleMetric.READY)) == 1,
            "first instance ready",
        )
        control.request_restart("test.metrics_restart")
        await _wait_until(
            lambda: len(_metrics(publisher, HostedApplicationLifecycleMetric.READY)) == 2,
            "second instance ready",
        )
        control.request_shutdown("test.done")
        await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("test.abort")
            await asyncio.gather(run, return_exceptions=True)

    restarts = _metrics(publisher, HostedApplicationLifecycleMetric.RESTART_COUNT)
    assert [item["value"] for item in restarts] == [1]
    exit_classes = _metrics(publisher, HostedApplicationLifecycleMetric.EXIT_CLASS)
    assert [item["label"] for item in exit_classes] == ["restart_requested", "clean_stop"]
    assert len(_metrics(publisher, HostedApplicationLifecycleMetric.START)) == 2


async def test_r009_forced_exit_class_matches_the_supervisor_classification(tmp_path: Path) -> None:
    controller = AsyncioActiveWork()
    profile = make_profile(
        runtime=JournalRuntime(),
        restart=RestartPolicy.never(),
        shutdown=ShutdownPolicy.drain_then_cancel(
            drain_timeout_seconds=0.2, cancel_timeout_seconds=1.0, flush_timeout_seconds=1.0
        ),
    )
    definition = definition_of(profile)
    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        engine, _, _ = build_engine(
            launch.definition,
            tmp_path,
            publisher=publisher,
            control=launch.control,
            instance_id=launch.instance_id,
            active_work_controller=controller,
        )
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: "forced-attempt-0",
    )
    run = asyncio.ensure_future(supervisor.run())
    try:
        await _wait_until(
            lambda: len(_metrics(publisher, HostedApplicationLifecycleMetric.READY)) == 1,
            "instance ready",
        )

        async def stuck() -> None:
            await asyncio.sleep(30.0)

        controller.submit("stuck", stuck)
        control.request_shutdown("test.forced")
        result = await asyncio.wait_for(run, timeout=_WAIT_SECONDS)
    finally:
        if not run.done():
            control.request_shutdown("test.abort")
            await asyncio.gather(run, return_exceptions=True)

    exit_classes = _metrics(publisher, HostedApplicationLifecycleMetric.EXIT_CLASS)
    assert [item["label"] for item in exit_classes] == ["forced_termination"]
    assert result.final_exit.exit_kind.value == "forced_termination"


async def test_r009_failed_run_reports_its_exit_class_before_the_failure_event(tmp_path: Path) -> None:
    journal: list[str] = []
    profile = make_profile(
        runtime=JournalRuntime(journal),
        restart=RestartPolicy.never(),
        components=(
            HostedApplicationComponentRegistration(
                component=OrderedComponent("core", journal, fail_start=True), required=True
            ),
        ),
    )
    definition = definition_of(profile)
    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        engine, _, _ = build_engine(
            launch.definition,
            tmp_path,
            publisher=publisher,
            control=launch.control,
            instance_id=launch.instance_id,
        )
        return engine

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: "failed-attempt-0",
    )
    result = await asyncio.wait_for(supervisor.run(), timeout=_WAIT_SECONDS)

    exit_classes = _metrics(publisher, HostedApplicationLifecycleMetric.EXIT_CLASS)
    assert [item["label"] for item in exit_classes] == ["startup_failure"]
    # The metric label is the supervisor's own classification of the same attempt.
    assert result.final_exit.exit_kind.value == exit_classes[0]["label"]
    types = [(event.event_type, event.payload.get("metric")) for event in publisher.events]
    metric_index = types.index((HostedApplicationEventType.LIFECYCLE_METRIC, "exit_class"))
    failed_index = types.index((HostedApplicationEventType.APPLICATION_FAILED, None))
    assert metric_index < failed_index == len(types) - 1


async def test_r009_pre_engine_failure_reports_exit_class_before_the_failure_event() -> None:
    profile = make_profile(runtime=JournalRuntime(), restart=RestartPolicy.never())
    definition = definition_of(profile)
    clock = SystemClock()
    control = HostedApplicationControlCoordinator(clock=clock)
    publisher = RecordingPublisher()

    def engine_factory(launch: HostedApplicationSupervisorLaunchContext):
        raise RuntimeError("engine factory is broken")

    supervisor = HostedApplicationSupervisor(
        definition=definition,
        engine_factory=engine_factory,
        control=control,
        event_publisher=publisher,
        clock=clock,
        instance_id_generator=lambda: "factory-attempt-0",
    )
    result = await asyncio.wait_for(supervisor.run(), timeout=_WAIT_SECONDS)

    exit_classes = _metrics(publisher, HostedApplicationLifecycleMetric.EXIT_CLASS)
    assert [item["label"] for item in exit_classes] == ["supervisor_error"]
    assert result.final_exit.exit_kind.value == "supervisor_error"
    last = publisher.events[-1]
    assert last.event_type is HostedApplicationEventType.APPLICATION_FAILED


async def test_r009_metrics_reach_the_canonical_observability_export_bridge(tmp_path: Path) -> None:
    exporter = InMemoryObservabilityExporter()
    bridge = ObservabilityHostedApplicationEventPublisher(exporter, policy=ObservabilityExportPolicy(enabled=True))

    class _Fanout:
        def __init__(self) -> None:
            self.recorded = RecordingPublisher()

        async def publish(self, event) -> None:
            await self.recorded.publish(event)
            await bridge.publish(event)

    fanout = _Fanout()
    profile = make_profile(runtime=JournalRuntime())
    engine, _, _ = build_engine(definition_of(profile), tmp_path, publisher=fanout)  # type: ignore[arg-type]
    await engine.start()
    await engine.stop(reason_code="test.export")

    exported = [
        envelope.sanitized_application_attributes.attributes
        for envelope in exporter.envelopes
        if envelope.event_type == HostedApplicationEventType.LIFECYCLE_METRIC.value
        and envelope.sanitized_application_attributes is not None
    ]
    names = {str(attributes["hosting.metric_name"]) for attributes in exported}
    assert {"start", "ready", "drain_duration"} <= names
    ready = next(attributes for attributes in exported if attributes["hosting.metric_name"] == "ready")
    assert ready["hosting.metric_unit"] == "seconds"
    assert isinstance(ready["hosting.metric_value"], float)
    assert ready["hosting.application_id"] == "pdc_d4_app"


def test_r009_hosting_owns_no_metrics_backend() -> None:
    """The metrics module only builds events; it holds no store, registry or exporter."""
    source = inspect.getsource(hosting_metrics)
    for forbidden in (
        "prometheus",
        "statsd",
        "opentelemetry",
        "sqlite",
        "open(",
        "threading",
        "Counter(",
        "Histogram(",
    ):
        assert forbidden not in source, forbidden
    public_callables = [
        name
        for name, value in vars(hosting_metrics).items()
        if not name.startswith("_") and inspect.isclass(value) and value.__module__ == hosting_metrics.__name__
    ]
    # Only the metric-name enum is a class; there is no emitter/registry/store type.
    assert public_callables == ["HostedApplicationLifecycleMetric"]
