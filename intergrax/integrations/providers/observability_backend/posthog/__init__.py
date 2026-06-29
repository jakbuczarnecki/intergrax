# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "POSTHOG_OBSERVABILITY_PROVIDER_ID",
    "PosthogObservabilityIntegration",
    "PosthogObservabilityIntegrationConfig",
    "PosthogObservabilityTransport",
    "create_posthog_observability_backend",
    "create_posthog_observability_integration",
    "register_posthog_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_posthog_observability_backend",
        "create_posthog_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "POSTHOG_OBSERVABILITY_PROVIDER_ID",
        "PosthogObservabilityIntegration",
        "PosthogObservabilityIntegrationConfig",
        "PosthogObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_posthog_integration":
        from intergrax.integrations.providers.observability_backend.posthog.register import (
            register_posthog_integration,
        )

        return register_posthog_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.posthog import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.posthog import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
