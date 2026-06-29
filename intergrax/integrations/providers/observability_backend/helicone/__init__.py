# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "HELICONE_OBSERVABILITY_PROVIDER_ID",
    "HeliconeObservabilityIntegration",
    "HeliconeObservabilityIntegrationConfig",
    "HeliconeObservabilityTransport",
    "create_helicone_observability_backend",
    "create_helicone_observability_integration",
    "register_helicone_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_helicone_observability_backend",
        "create_helicone_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "HELICONE_OBSERVABILITY_PROVIDER_ID",
        "HeliconeObservabilityIntegration",
        "HeliconeObservabilityIntegrationConfig",
        "HeliconeObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_helicone_integration":
        from intergrax.integrations.providers.observability_backend.helicone.register import (
            register_helicone_integration,
        )

        return register_helicone_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.helicone import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.helicone import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
