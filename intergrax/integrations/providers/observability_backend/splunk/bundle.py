# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend

from intergrax.integrations._shared.p7.factories import create_splunk_observability_backend as _legacy_create_splunk_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.splunk.integration import (
    SPLUNK_OBSERVABILITY_PROVIDER_ID,
    SPLUNK_SUPPORTED_SIGNALS,
    SplunkObservabilityIntegration,
    SplunkObservabilityIntegrationConfig,
    SplunkObservabilityTransport,
)

__all__ = [
    "create_splunk_observability_backend",
    "create_splunk_observability_integration",
]


def create_splunk_observability_integration(
    *,
    transport: SplunkObservabilityTransport | None = None,
    enabled: bool = False,
) -> SplunkObservabilityIntegration:
    """
    Build a contract-based Splunk observability vendor integration.

    The legacy query facade (create_splunk_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Splunk observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return SplunkObservabilityIntegration.from_transport(transport, enabled=enabled)
    return SplunkObservabilityIntegration.for_provider(
        provider_id=SPLUNK_OBSERVABILITY_PROVIDER_ID,
        supported_signals=SPLUNK_SUPPORTED_SIGNALS,
        display_name="Splunk",
        config=SplunkObservabilityIntegrationConfig(enabled=enabled),
    )


def create_splunk_observability_backend(
    *,
    observability_backend: ObservabilityBackend | None = None,
    client: object | None = None,
    client_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> SplunkObservabilityIntegration:
    """Compatibility shim — constructs SplunkObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_splunk_observability_backend(
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )
    if isinstance(runtime, SplunkObservabilityIntegration):
        return runtime
    return SplunkObservabilityIntegration.from_client(runtime)
