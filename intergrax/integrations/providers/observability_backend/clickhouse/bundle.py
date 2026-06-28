# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p3.factories import create_clickhouse_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.clickhouse.integration import (
    CLICKHOUSE_OBSERVABILITY_PROVIDER_ID,
    CLICKHOUSE_SUPPORTED_SIGNALS,
    ClickhouseObservabilityIntegration,
    ClickhouseObservabilityIntegrationConfig,
    ClickhouseObservabilityTransport,
)

__all__ = [
    "create_clickhouse_observability_backend",
    "create_clickhouse_observability_integration",
]


def create_clickhouse_observability_integration(
    *,
    transport: ClickhouseObservabilityTransport | None = None,
    enabled: bool = False,
) -> ClickhouseObservabilityIntegration:
    """
    Build a contract-based Clickhouse observability vendor integration.

    The legacy query facade (create_clickhouse_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Clickhouse observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return ClickhouseObservabilityIntegration.from_transport(transport, enabled=enabled)
    return ClickhouseObservabilityIntegration.for_provider(
        provider_id=CLICKHOUSE_OBSERVABILITY_PROVIDER_ID,
        supported_signals=CLICKHOUSE_SUPPORTED_SIGNALS,
        display_name="Clickhouse",
        config=ClickhouseObservabilityIntegrationConfig(enabled=enabled),
    )
