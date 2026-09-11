# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""W5-B2 / W5-C — explicit composition-root wiring for bounded runtime event delivery."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.event_delivery import EventDeliveryPolicy, EventExportSinkPort, EventSinkPort
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InternalDeliveryMetrics,
    NoopEventExportSink,
    RuntimeEventExportSink,
)


@dataclass(frozen=True, slots=True)
class ApplicationRuntimeEventDeliveryWiring:
    """Owned delivery stack for one application environment (no globals)."""

    policy: EventDeliveryPolicy | None
    event_export_sink: EventExportSinkPort | None
    export_bridge: RuntimeEventExportSink | None
    downstream_sink: EventSinkPort | None
    bounded_sink: BoundedEventSink | None
    delivery_metrics: InternalDeliveryMetrics | None

    @classmethod
    def disabled(cls) -> ApplicationRuntimeEventDeliveryWiring:
        return cls(
            policy=None,
            event_export_sink=None,
            export_bridge=None,
            downstream_sink=None,
            bounded_sink=None,
            delivery_metrics=None,
        )


def resolve_application_runtime_event_delivery_wiring(
    env: ApplicationEnvironmentProfile,
) -> ApplicationRuntimeEventDeliveryWiring:
    profile = env.observability_profile
    if not profile.bounded_event_delivery_enabled:
        return ApplicationRuntimeEventDeliveryWiring.disabled()
    policy = EventDeliveryPolicy(
        max_capacity=profile.bounded_event_delivery_max_capacity,
        important_wait_timeout_seconds=profile.bounded_event_delivery_important_wait_timeout_seconds,
    )
    metrics = InternalDeliveryMetrics()
    event_export_sink = NoopEventExportSink()
    export_bridge = RuntimeEventExportSink(event_export_sink, delivery_metrics=metrics)
    bounded = BoundedEventSink(export_bridge, policy)
    return ApplicationRuntimeEventDeliveryWiring(
        policy=policy,
        event_export_sink=event_export_sink,
        export_bridge=export_bridge,
        downstream_sink=export_bridge,
        bounded_sink=bounded,
        delivery_metrics=metrics,
    )


def compose_runtime_event_bus(
    delivery_wiring: ApplicationRuntimeEventDeliveryWiring,
    *,
    record_history: bool = True,
) -> RuntimeEventBus:
    event_sink = delivery_wiring.bounded_sink
    return RuntimeEventBus(
        record_history=record_history,
        event_sink=event_sink,
        delivery_metrics=delivery_wiring.delivery_metrics,
    )


def close_application_runtime_event_delivery(
    wiring: ApplicationRuntimeEventDeliveryWiring,
    *,
    event_bus: RuntimeEventBus | None = None,
) -> None:
    """Shutdown delivery stack: bus first (closes bounded sink + export pipeline)."""
    if event_bus is not None and not event_bus.closed:
        event_bus.close()
        return
    if wiring.bounded_sink is not None and not wiring.bounded_sink.closed:
        wiring.bounded_sink.close()
