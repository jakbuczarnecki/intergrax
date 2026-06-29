# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "PHOENIX_OBSERVABILITY_PROVIDER_ID",
    "PhoenixObservabilityIntegration",
    "PhoenixObservabilityIntegrationConfig",
    "PhoenixObservabilityTransport",
    "create_phoenix_observability_backend",
    "create_phoenix_observability_integration",
    "register_phoenix_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_phoenix_observability_backend",
        "create_phoenix_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "PHOENIX_OBSERVABILITY_PROVIDER_ID",
        "PhoenixObservabilityIntegration",
        "PhoenixObservabilityIntegrationConfig",
        "PhoenixObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_phoenix_integration":
        from intergrax.integrations.providers.observability_backend.phoenix.register import (
            register_phoenix_integration,
        )

        return register_phoenix_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.phoenix import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.phoenix import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
