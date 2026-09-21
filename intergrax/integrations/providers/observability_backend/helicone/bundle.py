# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend

from intergrax.integrations._shared.p4.factories import create_helicone_observability_backend as _legacy_create_helicone_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.helicone.integration import (
    HELICONE_OBSERVABILITY_PROVIDER_ID,
    HELICONE_SUPPORTED_SIGNALS,
    HeliconeObservabilityIntegration,
    HeliconeObservabilityIntegrationConfig,
    HeliconeObservabilityTransport,
)

__all__ = [
    "create_helicone_observability_backend",
    "create_helicone_observability_integration",
]


def create_helicone_observability_integration(
    *,
    transport: HeliconeObservabilityTransport | None = None,
    enabled: bool = False,
) -> HeliconeObservabilityIntegration:
    """
    Build a contract-based Helicone observability vendor integration.

    The legacy query facade (create_helicone_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Helicone observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return HeliconeObservabilityIntegration.from_transport(transport, enabled=enabled)
    return HeliconeObservabilityIntegration.for_provider(
        provider_id=HELICONE_OBSERVABILITY_PROVIDER_ID,
        supported_signals=HELICONE_SUPPORTED_SIGNALS,
        display_name="Helicone",
        config=HeliconeObservabilityIntegrationConfig(enabled=enabled),
    )


def create_helicone_observability_backend(
    *,
    observability_backend: ObservabilityBackend | None = None,
    client: object | None = None,
    client_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> HeliconeObservabilityIntegration:
    """Compatibility shim — constructs HeliconeObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_helicone_observability_backend(
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )
    if isinstance(runtime, HeliconeObservabilityIntegration):
        return runtime
    return HeliconeObservabilityIntegration.from_client(runtime)
