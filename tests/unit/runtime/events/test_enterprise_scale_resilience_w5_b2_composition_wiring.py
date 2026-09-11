# © Artur Czarnecki. All rights reserved.

"""W5-B2 — production composition wiring for bounded runtime event delivery."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.harness_host_runtime import (
    build_harness_host_runtime,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ObservabilityProfile,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.event_delivery import (
    AcceptingObservabilityEventSink,
    BoundedEventSink,
)
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.runtime_events import runtime_event_test_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _bounded_delivery_environment(
    profile_id: str = "w5.b2.prod",
) -> ApplicationEnvironmentProfile:
    """Production-style bounded delivery without strict host persistence prerequisites."""
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    env.observability_profile = env.observability_profile.model_copy(
        update={"bounded_event_delivery_enabled": True},
    )
    return env


def _terminal_event() -> RuntimeEvent:
    return RuntimeEvent(
        event_type=RuntimeEventType.TASK_COMPLETED,
        phase=ExecutionPhase.COMPLETION,
        **runtime_event_test_identity(),
    )


def test_production_composition_creates_bounded_sink() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = _bounded_delivery_environment()
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    bus = runtime.env_wiring.build_context.runtime_event_bus
    assert bus is not None
    assert bus.event_sink is not None
    assert isinstance(bus.event_sink, BoundedEventSink)
    runtime.close()


@pytest.mark.asyncio
async def test_published_event_reaches_bounded_sink() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = _bounded_delivery_environment()
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    bus = runtime.env_wiring.build_context.runtime_event_bus
    assert bus is not None
    delivery = runtime.env_wiring.event_delivery
    downstream = delivery.downstream_sink
    assert isinstance(downstream, AcceptingObservabilityEventSink)
    await bus.publish(_terminal_event())
    metrics = bus.delivery_metrics
    assert metrics is not None
    assert metrics.snapshot().events_accepted >= 1
    runtime.close()
    assert downstream.accepted_count >= 1


def test_shutdown_ordering_closes_bus_and_sink() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = _bounded_delivery_environment()
    runtime = build_harness_host_runtime(manifest, env, settings=settings)
    bus = runtime.env_wiring.build_context.runtime_event_bus
    assert bus is not None
    bounded = runtime.env_wiring.event_delivery.bounded_sink
    assert bounded is not None
    runtime.close()
    assert bus.closed
    assert bounded.closed
    assert not bounded._worker.is_alive()  # noqa: SLF001 — qualification shutdown


def test_no_singleton_leakage_between_environments() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env_a = _bounded_delivery_environment(profile_id="w5.b2.a")
    env_b = _bounded_delivery_environment(profile_id="w5.b2.b")
    wiring_a = wire_application_environment(manifest, env_a, conformance_check=False)
    wiring_b = wire_application_environment(manifest, env_b, conformance_check=False)
    sink_a = wiring_a.event_delivery.bounded_sink
    sink_b = wiring_b.event_delivery.bounded_sink
    assert sink_a is not None
    assert sink_b is not None
    assert sink_a is not sink_b


def test_legacy_composition_without_sink_preserves_behavior() -> None:
    settings = LabApplicationSettings.from_env()
    manifest = build_lab_manifest(settings)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="w5.b2.legacy")
    assert env.observability_profile.bounded_event_delivery_enabled is False
    wiring = wire_application_environment(manifest, env, conformance_check=False)
    bus = wiring.build_context.runtime_event_bus
    assert isinstance(bus, RuntimeEventBus)
    assert bus.event_sink is None
    assert bus.delivery_metrics is None
