# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend

from intergrax.integrations._shared.p3.factories import create_sentry_observability_backend as _legacy_create_sentry_observability_backend
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.observability_backend.sentry.client import (
    SentryCaptureClient,
    open_sentry_sdk_capture_client,
)
from intergrax.integrations.providers.observability_backend.sentry.config import SentryIntegrationConfig
from intergrax.integrations.providers.observability_backend.sentry.integration import (
    SENTRY_OBSERVABILITY_PROVIDER_ID,
    SENTRY_SUPPORTED_SIGNALS,
    SentryObservabilityIntegration,
    SentryObservabilityIntegrationConfig,
    SentryObservabilityTransport,
)
from intergrax.integrations.providers.observability_backend.sentry.transport import (
    SentrySdkObservabilityTransport,
    map_vendor_payload_to_sentry_event,
)

__all__ = [
    "SentryCaptureClient",
    "SentryIntegrationConfig",
    "SentrySdkObservabilityTransport",
    "create_sentry_observability_backend",
    "create_sentry_observability_integration",
    "create_sentry_observability_transport",
    "map_vendor_payload_to_sentry_event",
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


def create_sentry_observability_backend(
    *,
    observability_backend: ObservabilityBackend | None = None,
    client: object | None = None,
    client_factory: Callable[[], object] | None = None,
    **config_overrides: object,
) -> SentryObservabilityIntegration:
    """Compatibility shim — constructs SentryObservabilityIntegration from legacy runtime."""
    runtime = _legacy_create_sentry_observability_backend(
        observability_backend=observability_backend,
        client=client,
        client_factory=client_factory,
        **config_overrides,
    )
    if isinstance(runtime, SentryObservabilityIntegration):
        return runtime
    return SentryObservabilityIntegration.from_client(runtime)


def create_sentry_observability_transport(
    *,
    client: SentryCaptureClient | None = None,
    flush_after_capture: bool = False,
    **config_overrides: object,
) -> SentrySdkObservabilityTransport:
    """Build the concrete Sentry observability export transport."""
    if client is not None:
        return SentrySdkObservabilityTransport(
            client,
            flush_after_capture=flush_after_capture,
        )
    config = SentryIntegrationConfig.from_env(**config_overrides)
    if not config.dsn:
        raise IntegrationConfigurationError(
            "Sentry observability transport requires a DSN in provider configuration",
        )
    sdk_client = open_sentry_sdk_capture_client(config)
    return SentrySdkObservabilityTransport(
        sdk_client,
        flush_after_capture=flush_after_capture,
        flush_timeout=config.shutdown_timeout_seconds,
    )
