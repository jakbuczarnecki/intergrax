# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p3.factories import create_datadog_observability_backend as _legacy_create_datadog_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.datadog.integration import (
    DATADOG_OBSERVABILITY_PROVIDER_ID,
    DATADOG_SUPPORTED_SIGNALS,
    DatadogObservabilityIntegration,
    DatadogObservabilityIntegrationConfig,
    DatadogObservabilityTransport,
)

__all__ = [
    "create_datadog_observability_backend",
    "create_datadog_observability_integration",
]


def create_datadog_observability_integration(
    *,
    transport: DatadogObservabilityTransport | None = None,
    enabled: bool = False,
) -> DatadogObservabilityIntegration:
    """
    Build a contract-based Datadog observability vendor integration.

    The legacy query facade (create_datadog_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Datadog observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return DatadogObservabilityIntegration.from_transport(transport, enabled=enabled)
    return DatadogObservabilityIntegration.for_provider(
        provider_id=DATADOG_OBSERVABILITY_PROVIDER_ID,
        supported_signals=DATADOG_SUPPORTED_SIGNALS,
        display_name="Datadog",
        config=DatadogObservabilityIntegrationConfig(enabled=enabled),
    )


def create_datadog_observability_backend(**kwargs: object) -> DatadogObservabilityIntegration:
    """Compatibility shim — constructs DatadogObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_datadog_observability_backend(**kwargs)
    if isinstance(runtime, DatadogObservabilityIntegration):
        return runtime
    return DatadogObservabilityIntegration.from_client(runtime)
