# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.observability_backend.sentry.config import (
    ENV_SENTRY_DEBUG,
    ENV_SENTRY_DSN,
    ENV_SENTRY_ENVIRONMENT,
    ENV_SENTRY_RELEASE,
    ENV_SENTRY_SERVER_NAME,
    ENV_SENTRY_SHUTDOWN_TIMEOUT_SECONDS,
    SentryIntegrationConfig,
)
from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "ENV_SENTRY_DEBUG",
    "ENV_SENTRY_DSN",
    "ENV_SENTRY_ENVIRONMENT",
    "ENV_SENTRY_RELEASE",
    "ENV_SENTRY_SERVER_NAME",
    "ENV_SENTRY_SHUTDOWN_TIMEOUT_SECONDS",
    "SENTRY_OBSERVABILITY_PROVIDER_ID",
    "SentryCaptureClient",
    "SentryIntegrationConfig",
    "SentryObservabilityIntegration",
    "SentryObservabilityIntegrationConfig",
    "SentryObservabilityTransport",
    "SentrySdkObservabilityTransport",
    "create_sentry_observability_backend",
    "create_sentry_observability_integration",
    "create_sentry_observability_transport",
    "map_vendor_payload_to_sentry_event",
    "register_sentry_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "SentryCaptureClient",
        "SentrySdkObservabilityTransport",
        "create_sentry_observability_backend",
        "create_sentry_observability_integration",
        "create_sentry_observability_transport",
        "map_vendor_payload_to_sentry_event",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SENTRY_OBSERVABILITY_PROVIDER_ID",
        "SentryObservabilityIntegration",
        "SentryObservabilityIntegrationConfig",
        "SentryObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_sentry_integration":
        from intergrax.integrations.providers.observability_backend.sentry.register import (
            register_sentry_integration,
        )

        return register_sentry_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.sentry import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.sentry import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
