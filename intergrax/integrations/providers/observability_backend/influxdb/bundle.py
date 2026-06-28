# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p5.factories import create_influxdb_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.influxdb.integration import (
    INFLUXDB_OBSERVABILITY_PROVIDER_ID,
    INFLUXDB_SUPPORTED_SIGNALS,
    InfluxdbObservabilityIntegration,
    InfluxdbObservabilityIntegrationConfig,
    InfluxdbObservabilityTransport,
)

__all__ = [
    "create_influxdb_observability_backend",
    "create_influxdb_observability_integration",
]


def create_influxdb_observability_integration(
    *,
    transport: InfluxdbObservabilityTransport | None = None,
    enabled: bool = False,
) -> InfluxdbObservabilityIntegration:
    """
    Build a contract-based Influxdb observability vendor integration.

    The legacy query facade (create_influxdb_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Influxdb observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return InfluxdbObservabilityIntegration.from_transport(transport, enabled=enabled)
    return InfluxdbObservabilityIntegration.for_provider(
        provider_id=INFLUXDB_OBSERVABILITY_PROVIDER_ID,
        supported_signals=INFLUXDB_SUPPORTED_SIGNALS,
        display_name="Influxdb",
        config=InfluxdbObservabilityIntegrationConfig(enabled=enabled),
    )
