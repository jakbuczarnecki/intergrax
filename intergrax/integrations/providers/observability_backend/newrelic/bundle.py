# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p7.factories import create_newrelic_observability_backend as _legacy_create_newrelic_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.newrelic.integration import (
    NEWRELIC_OBSERVABILITY_PROVIDER_ID,
    NEWRELIC_SUPPORTED_SIGNALS,
    NewRelicObservabilityIntegration,
    NewRelicObservabilityIntegrationConfig,
    NewRelicObservabilityTransport,
)

__all__ = [
    "create_newrelic_observability_backend",
    "create_newrelic_observability_integration",
]


def create_newrelic_observability_integration(
    *,
    transport: NewRelicObservabilityTransport | None = None,
    enabled: bool = False,
) -> NewRelicObservabilityIntegration:
    """
    Build a contract-based New Relic observability vendor integration.

    The legacy query facade (create_newrelic_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "New Relic observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return NewRelicObservabilityIntegration.from_transport(transport, enabled=enabled)
    return NewRelicObservabilityIntegration.for_provider(
        provider_id=NEWRELIC_OBSERVABILITY_PROVIDER_ID,
        supported_signals=NEWRELIC_SUPPORTED_SIGNALS,
        display_name="New Relic",
        config=NewRelicObservabilityIntegrationConfig(enabled=enabled),
    )


def create_newrelic_observability_backend(**kwargs: object) -> NewRelicObservabilityIntegration:
    """Compatibility shim — constructs NewRelicObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_newrelic_observability_backend(**kwargs)
    if isinstance(runtime, NewRelicObservabilityIntegration):
        return runtime
    return NewRelicObservabilityIntegration.from_backend(runtime)
