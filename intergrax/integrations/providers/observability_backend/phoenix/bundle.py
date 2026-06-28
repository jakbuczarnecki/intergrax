# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p4.factories import create_phoenix_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.phoenix.integration import (
    PHOENIX_OBSERVABILITY_PROVIDER_ID,
    PHOENIX_SUPPORTED_SIGNALS,
    PhoenixObservabilityIntegration,
    PhoenixObservabilityIntegrationConfig,
    PhoenixObservabilityTransport,
)

__all__ = [
    "create_phoenix_observability_backend",
    "create_phoenix_observability_integration",
]


def create_phoenix_observability_integration(
    *,
    transport: PhoenixObservabilityTransport | None = None,
    enabled: bool = False,
) -> PhoenixObservabilityIntegration:
    """
    Build a contract-based Phoenix observability vendor integration.

    The legacy query facade (create_phoenix_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Phoenix observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return PhoenixObservabilityIntegration.from_transport(transport, enabled=enabled)
    return PhoenixObservabilityIntegration.for_provider(
        provider_id=PHOENIX_OBSERVABILITY_PROVIDER_ID,
        supported_signals=PHOENIX_SUPPORTED_SIGNALS,
        display_name="Phoenix",
        config=PhoenixObservabilityIntegrationConfig(enabled=enabled),
    )
