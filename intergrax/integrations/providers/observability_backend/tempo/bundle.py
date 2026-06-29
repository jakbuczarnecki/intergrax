# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p5.factories import create_tempo_observability_backend as _legacy_create_tempo_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.tempo.integration import (
    TEMPO_OBSERVABILITY_PROVIDER_ID,
    TEMPO_SUPPORTED_SIGNALS,
    TempoObservabilityIntegration,
    TempoObservabilityIntegrationConfig,
    TempoObservabilityTransport,
)

__all__ = [
    "create_tempo_observability_backend",
    "create_tempo_observability_integration",
]


def create_tempo_observability_integration(
    *,
    transport: TempoObservabilityTransport | None = None,
    enabled: bool = False,
) -> TempoObservabilityIntegration:
    """
    Build a contract-based Tempo observability vendor integration.

    The legacy query facade (create_tempo_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Tempo observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return TempoObservabilityIntegration.from_transport(transport, enabled=enabled)
    return TempoObservabilityIntegration.for_provider(
        provider_id=TEMPO_OBSERVABILITY_PROVIDER_ID,
        supported_signals=TEMPO_SUPPORTED_SIGNALS,
        display_name="Tempo",
        config=TempoObservabilityIntegrationConfig(enabled=enabled),
    )


def create_tempo_observability_backend(**kwargs: object) -> TempoObservabilityIntegration:
    """Compatibility shim — constructs TempoObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_tempo_observability_backend(**kwargs)
    if isinstance(runtime, TempoObservabilityIntegration):
        return runtime
    return TempoObservabilityIntegration.from_client(runtime)
