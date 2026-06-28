# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p5.factories import create_grafana_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.grafana.integration import (
    GRAFANA_OBSERVABILITY_PROVIDER_ID,
    GRAFANA_SUPPORTED_SIGNALS,
    GrafanaObservabilityIntegration,
    GrafanaObservabilityIntegrationConfig,
    GrafanaObservabilityTransport,
)

__all__ = [
    "create_grafana_observability_backend",
    "create_grafana_observability_integration",
]


def create_grafana_observability_integration(
    *,
    transport: GrafanaObservabilityTransport | None = None,
    enabled: bool = False,
) -> GrafanaObservabilityIntegration:
    """
    Build a contract-based Grafana observability vendor integration.

    The legacy query facade (create_grafana_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Grafana observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return GrafanaObservabilityIntegration.from_transport(transport, enabled=enabled)
    return GrafanaObservabilityIntegration.for_provider(
        provider_id=GRAFANA_OBSERVABILITY_PROVIDER_ID,
        supported_signals=GRAFANA_SUPPORTED_SIGNALS,
        display_name="Grafana",
        config=GrafanaObservabilityIntegrationConfig(enabled=enabled),
    )
