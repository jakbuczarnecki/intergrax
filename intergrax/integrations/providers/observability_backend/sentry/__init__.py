# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SENTRY_OBSERVABILITY_PROVIDER_ID",
    "SentryObservabilityIntegration",
    "SentryObservabilityIntegrationConfig",
    "SentryObservabilityTransport",
    "create_sentry_observability_backend",
    "create_sentry_observability_integration",
    "register_sentry_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_sentry_observability_backend",
        "create_sentry_observability_integration",
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
