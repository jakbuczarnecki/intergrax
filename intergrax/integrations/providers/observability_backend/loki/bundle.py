# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p5.factories import create_loki_observability_backend as _legacy_create_loki_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.loki.integration import (
    LOKI_OBSERVABILITY_PROVIDER_ID,
    LOKI_SUPPORTED_SIGNALS,
    LokiObservabilityIntegration,
    LokiObservabilityIntegrationConfig,
    LokiObservabilityTransport,
)

__all__ = [
    "create_loki_observability_backend",
    "create_loki_observability_integration",
]


def create_loki_observability_integration(
    *,
    transport: LokiObservabilityTransport | None = None,
    enabled: bool = False,
) -> LokiObservabilityIntegration:
    """
    Build a contract-based Loki observability vendor integration.

    The legacy query facade (create_loki_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Loki observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return LokiObservabilityIntegration.from_transport(transport, enabled=enabled)
    return LokiObservabilityIntegration.for_provider(
        provider_id=LOKI_OBSERVABILITY_PROVIDER_ID,
        supported_signals=LOKI_SUPPORTED_SIGNALS,
        display_name="Loki",
        config=LokiObservabilityIntegrationConfig(enabled=enabled),
    )


def create_loki_observability_backend(**kwargs: object) -> LokiObservabilityIntegration:
    """Compatibility shim — constructs LokiObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_loki_observability_backend(**kwargs)
    if isinstance(runtime, LokiObservabilityIntegration):
        return runtime
    return LokiObservabilityIntegration.from_backend(runtime)
