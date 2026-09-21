# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend

from intergrax.integrations._shared.p4.factories import create_arize_observability_backend as _legacy_create_arize_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.arize.integration import (
    ARIZE_OBSERVABILITY_PROVIDER_ID,
    ARIZE_SUPPORTED_SIGNALS,
    ArizeObservabilityIntegration,
    ArizeObservabilityIntegrationConfig,
    ArizeObservabilityTransport,
)

__all__ = [
    "create_arize_observability_backend",
    "create_arize_observability_integration",
]


def create_arize_observability_integration(
    *,
    transport: ArizeObservabilityTransport | None = None,
    enabled: bool = False,
) -> ArizeObservabilityIntegration:
    """
    Build a contract-based Arize observability vendor integration.

    The legacy query facade (create_arize_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Arize observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return ArizeObservabilityIntegration.from_transport(transport, enabled=enabled)
    return ArizeObservabilityIntegration.for_provider(
        provider_id=ARIZE_OBSERVABILITY_PROVIDER_ID,
        supported_signals=ARIZE_SUPPORTED_SIGNALS,
        display_name="Arize",
        config=ArizeObservabilityIntegrationConfig(enabled=enabled),
    )


def create_arize_observability_backend(
    *,
    observability_backend: ObservabilityBackend | None = None,
    client: object | None = None,
    client_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> ArizeObservabilityIntegration:
    """Compatibility shim — constructs ArizeObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_arize_observability_backend(
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )
    if isinstance(runtime, ArizeObservabilityIntegration):
        return runtime
    return ArizeObservabilityIntegration.from_client(runtime)
