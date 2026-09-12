# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""W5-B2 / W5-C / W5-D — explicit composition-root wiring for bounded runtime event delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.event_delivery import EventDeliveryPolicy, EventExportSinkPort, EventSinkPort
from intergrax.contracts.observability_export import (
    ConfigurationError,
    EventExportSinkFactoryPort,
    ExporterKind,
    ObservabilityExportProfile,
    OtlpExportConfiguration,
    OtlpTransportPort,
)
from intergrax.runtime.observability.exporters.distributed.collector_transport import (
    CollectorTransport,
)
from intergrax.runtime.observability.exporters.distributed.distributed_configuration import (
    DistributedTransportConfiguration,
)
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.observability.event_delivery import (
    BoundedEventSink,
    InternalDeliveryMetrics,
    RuntimeEventExportSink,
)
from intergrax.runtime.observability.event_delivery.export_factory import (
    ObservabilityExportSinkFactory,
)


@runtime_checkable
class _ObservabilityOtlpEndpointSettings(Protocol):
    observability_otlp_endpoint: str
    observability_otlp_timeout_seconds: float


def _resolve_otlp_export_configuration(
    env: ApplicationEnvironmentProfile,
    *,
    settings: object | None,
) -> OtlpExportConfiguration | None:
    profile = env.observability_profile
    endpoint = profile.otlp_export_endpoint.strip()
    timeout_seconds = profile.otlp_export_timeout_seconds
    if settings is not None and isinstance(settings, _ObservabilityOtlpEndpointSettings):
        settings_endpoint = settings.observability_otlp_endpoint.strip()
        if settings_endpoint:
            endpoint = settings_endpoint
            timeout_seconds = settings.observability_otlp_timeout_seconds
    if not endpoint:
        return None
    return OtlpExportConfiguration(
        endpoint=endpoint,
        protocol=profile.otlp_export_protocol,
        timeout_seconds=timeout_seconds,
    )


def _resolve_distributed_transport_configuration(
    env: ApplicationEnvironmentProfile,
    *,
    settings: object | None,
) -> DistributedTransportConfiguration | None:
    otlp_config = _resolve_otlp_export_configuration(env, settings=settings)
    if otlp_config is None:
        return None
    service_name = env.observability_profile.observability_export_service_name.strip()
    if not service_name:
        return None
    return DistributedTransportConfiguration(
        endpoint=otlp_config.endpoint,
        protocol=otlp_config.protocol,
        service_name=service_name,
        timeout_seconds=otlp_config.timeout_seconds,
    )


def _create_export_transport(
    export_profile: ObservabilityExportProfile,
    env: ApplicationEnvironmentProfile,
    *,
    settings: object | None,
) -> OtlpTransportPort | None:
    if export_profile.exporter_kind is ExporterKind.OTLP:
        config = _resolve_otlp_export_configuration(env, settings=settings)
        if config is None:
            return None
        return OtlpTransport(config)
    if export_profile.exporter_kind is ExporterKind.DISTRIBUTED_OTLP:
        config = _resolve_distributed_transport_configuration(env, settings=settings)
        if config is None:
            raise ConfigurationError(
                "DISTRIBUTED_OTLP requires non-empty otlp_export_endpoint and "
                "observability_export_service_name",
            )
        return CollectorTransport(config)
    return None


def resolve_observability_export_profile(
    env: ApplicationEnvironmentProfile,
) -> ObservabilityExportProfile:
    profile = env.observability_profile
    if not profile.bounded_event_delivery_enabled:
        return ObservabilityExportProfile(
            enabled=False,
            exporter_kind=ExporterKind.NOOP,
        )
    return ObservabilityExportProfile(
        enabled=True,
        exporter_kind=profile.observability_exporter_kind,
    )


@dataclass(frozen=True, slots=True)
class ApplicationRuntimeEventDeliveryWiring:
    """Owned delivery stack for one application environment (no globals)."""

    export_profile: ObservabilityExportProfile | None
    policy: EventDeliveryPolicy | None
    event_export_sink: EventExportSinkPort | None
    export_bridge: RuntimeEventExportSink | None
    downstream_sink: EventSinkPort | None
    bounded_sink: BoundedEventSink | None
    delivery_metrics: InternalDeliveryMetrics | None
    otlp_transport: OtlpTransportPort | None = None

    @classmethod
    def disabled(cls) -> ApplicationRuntimeEventDeliveryWiring:
        return cls(
            export_profile=None,
            policy=None,
            event_export_sink=None,
            export_bridge=None,
            downstream_sink=None,
            bounded_sink=None,
            delivery_metrics=None,
            otlp_transport=None,
        )


def resolve_application_runtime_event_delivery_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    export_sink_factory: EventExportSinkFactoryPort | None = None,
    settings: object | None = None,
) -> ApplicationRuntimeEventDeliveryWiring:
    profile = env.observability_profile
    if not profile.bounded_event_delivery_enabled:
        return ApplicationRuntimeEventDeliveryWiring.disabled()
    export_profile = resolve_observability_export_profile(env)
    policy = EventDeliveryPolicy(
        max_capacity=profile.bounded_event_delivery_max_capacity,
        important_wait_timeout_seconds=profile.bounded_event_delivery_important_wait_timeout_seconds,
    )
    exporter_kind_label = export_profile.exporter_kind.value
    metrics = InternalDeliveryMetrics(exporter_kind=exporter_kind_label)
    otlp_transport = _create_export_transport(export_profile, env, settings=settings)
    factory = export_sink_factory or ObservabilityExportSinkFactory(
        otlp_transport=otlp_transport,
    )
    event_export_sink = factory.create(export_profile)
    export_bridge = RuntimeEventExportSink(event_export_sink, delivery_metrics=metrics)
    bounded = BoundedEventSink(export_bridge, policy)
    return ApplicationRuntimeEventDeliveryWiring(
        export_profile=export_profile,
        policy=policy,
        event_export_sink=event_export_sink,
        export_bridge=export_bridge,
        downstream_sink=export_bridge,
        bounded_sink=bounded,
        delivery_metrics=metrics,
        otlp_transport=otlp_transport,
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
