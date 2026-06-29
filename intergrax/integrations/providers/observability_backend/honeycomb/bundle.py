# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p4.factories import create_honeycomb_observability_backend as _legacy_create_honeycomb_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.honeycomb.integration import (
    HONEYCOMB_OBSERVABILITY_PROVIDER_ID,
    HONEYCOMB_SUPPORTED_SIGNALS,
    HoneycombObservabilityIntegration,
    HoneycombObservabilityIntegrationConfig,
    HoneycombObservabilityTransport,
)

__all__ = [
    "create_honeycomb_observability_backend",
    "create_honeycomb_observability_integration",
]


def create_honeycomb_observability_integration(
    *,
    transport: HoneycombObservabilityTransport | None = None,
    enabled: bool = False,
) -> HoneycombObservabilityIntegration:
    """
    Build a contract-based Honeycomb observability vendor integration.

    The legacy query facade (create_honeycomb_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Honeycomb observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return HoneycombObservabilityIntegration.from_transport(transport, enabled=enabled)
    return HoneycombObservabilityIntegration.for_provider(
        provider_id=HONEYCOMB_OBSERVABILITY_PROVIDER_ID,
        supported_signals=HONEYCOMB_SUPPORTED_SIGNALS,
        display_name="Honeycomb",
        config=HoneycombObservabilityIntegrationConfig(enabled=enabled),
    )


def create_honeycomb_observability_backend(**kwargs: object) -> HoneycombObservabilityIntegration:
    """Compatibility shim — constructs HoneycombObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_honeycomb_observability_backend(**kwargs)
    if isinstance(runtime, HoneycombObservabilityIntegration):
        return runtime
    return HoneycombObservabilityIntegration.from_client(runtime)
