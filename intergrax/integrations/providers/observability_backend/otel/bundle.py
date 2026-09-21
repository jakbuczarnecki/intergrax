# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend

from intergrax.integrations._shared.p2.factories import create_otel_observability_backend as _legacy_create_otel_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.otel.integration import (
    OTEL_OBSERVABILITY_PROVIDER_ID,
    OTEL_SUPPORTED_SIGNALS,
    OtelObservabilityIntegration,
    OtelObservabilityIntegrationConfig,
    OtelObservabilityTransport,
)

__all__ = [
    "create_otel_observability_backend",
    "create_otel_observability_integration",
]


def create_otel_observability_integration(
    *,
    transport: OtelObservabilityTransport | None = None,
    enabled: bool = False,
) -> OtelObservabilityIntegration:
    """
    Build a contract-based OTel observability vendor integration.

    The legacy query facade (create_otel_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "OTel observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return OtelObservabilityIntegration.from_transport(transport, enabled=enabled)
    return OtelObservabilityIntegration.for_provider(
        provider_id=OTEL_OBSERVABILITY_PROVIDER_ID,
        supported_signals=OTEL_SUPPORTED_SIGNALS,
        display_name="OTel",
        config=OtelObservabilityIntegrationConfig(enabled=enabled),
    )


def create_otel_observability_backend(
    *,
    observability_backend: ObservabilityBackend | None = None,
    exporter: object | None = None,
    exporter_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> OtelObservabilityIntegration:
    """Compatibility shim — constructs OtelObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_otel_observability_backend(
        observability_backend=observability_backend,
        exporter=exporter,
        exporter_factory=exporter_factory,
        **config_overrides,
    )
    if isinstance(runtime, OtelObservabilityIntegration):
        return runtime
    return OtelObservabilityIntegration.from_client(runtime)
