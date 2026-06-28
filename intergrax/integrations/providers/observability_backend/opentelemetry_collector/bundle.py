# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p6.factories import create_opentelemetry_collector_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.opentelemetry_collector.integration import (
    OPENTELEMETRY_COLLECTOR_OBSERVABILITY_PROVIDER_ID,
    OPENTELEMETRY_COLLECTOR_SUPPORTED_SIGNALS,
    OpenTelemetryCollectorObservabilityIntegration,
    OpenTelemetryCollectorObservabilityIntegrationConfig,
    OpenTelemetryCollectorObservabilityTransport,
)

__all__ = [
    "create_opentelemetry_collector_observability_backend",
    "create_opentelemetry_collector_observability_integration",
]


def create_opentelemetry_collector_observability_integration(
    *,
    transport: OpenTelemetryCollectorObservabilityTransport | None = None,
    enabled: bool = False,
) -> OpenTelemetryCollectorObservabilityIntegration:
    """
    Build a contract-based OpenTelemetry Collector observability vendor integration.

    The legacy query facade (create_opentelemetry_collector_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "OpenTelemetry Collector observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return OpenTelemetryCollectorObservabilityIntegration.from_transport(transport, enabled=enabled)
    return OpenTelemetryCollectorObservabilityIntegration.for_provider(
        provider_id=OPENTELEMETRY_COLLECTOR_OBSERVABILITY_PROVIDER_ID,
        supported_signals=OPENTELEMETRY_COLLECTOR_SUPPORTED_SIGNALS,
        display_name="OpenTelemetry Collector",
        config=OpenTelemetryCollectorObservabilityIntegrationConfig(enabled=enabled),
    )
