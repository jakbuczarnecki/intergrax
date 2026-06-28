# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "ARIZE_OBSERVABILITY_PROVIDER_ID",
    "ArizeObservabilityIntegration",
    "ArizeObservabilityIntegrationConfig",
    "ArizeObservabilityTransport",
    "create_arize_observability_backend",
    "create_arize_observability_integration",
    "register_arize_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_arize_observability_backend",
        "create_arize_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "ARIZE_OBSERVABILITY_PROVIDER_ID",
        "ArizeObservabilityIntegration",
        "ArizeObservabilityIntegrationConfig",
        "ArizeObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_arize_integration":
        from intergrax.integrations.providers.observability_backend.arize.register import (
            register_arize_integration,
        )

        return register_arize_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.arize import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.arize import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
