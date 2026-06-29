# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations._shared.p3.factories import create_sentry_observability_backend as _legacy_create_sentry_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.sentry.integration import (
    SENTRY_OBSERVABILITY_PROVIDER_ID,
    SENTRY_SUPPORTED_SIGNALS,
    SentryObservabilityIntegration,
    SentryObservabilityIntegrationConfig,
    SentryObservabilityTransport,
)

__all__ = [
    "create_sentry_observability_backend",
    "create_sentry_observability_integration",
]


def create_sentry_observability_integration(
    *,
    transport: SentryObservabilityTransport | None = None,
    enabled: bool = False,
) -> SentryObservabilityIntegration:
    """
    Build a contract-based Sentry observability vendor integration.

    The legacy query facade (create_sentry_observability_backend) is unchanged.
    Transport must be injected explicitly for enabled export; disabled by default.
    """
    if enabled and transport is None:
        raise IntegrationConfigurationError(
            "Sentry observability integration requires an injected transport when enabled=True",
        )
    if transport is not None:
        return SentryObservabilityIntegration.from_transport(transport, enabled=enabled)
    return SentryObservabilityIntegration.for_provider(
        provider_id=SENTRY_OBSERVABILITY_PROVIDER_ID,
        supported_signals=SENTRY_SUPPORTED_SIGNALS,
        display_name="Sentry",
        config=SentryObservabilityIntegrationConfig(enabled=enabled),
    )


def create_sentry_observability_backend(**kwargs: object) -> SentryObservabilityIntegration:
    """Compatibility shim — constructs SentryObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_sentry_observability_backend(**kwargs)
    if isinstance(runtime, SentryObservabilityIntegration):
        return runtime
    return SentryObservabilityIntegration.from_client(runtime)
