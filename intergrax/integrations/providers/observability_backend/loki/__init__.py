# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "LOKI_OBSERVABILITY_PROVIDER_ID",
    "LokiObservabilityIntegration",
    "LokiObservabilityIntegrationConfig",
    "LokiObservabilityTransport",
    "create_loki_observability_backend",
    "create_loki_observability_integration",
    "register_loki_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_loki_observability_backend",
        "create_loki_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "LOKI_OBSERVABILITY_PROVIDER_ID",
        "LokiObservabilityIntegration",
        "LokiObservabilityIntegrationConfig",
        "LokiObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_loki_integration":
        from intergrax.integrations.providers.observability_backend.loki.register import (
            register_loki_integration,
        )

        return register_loki_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.loki import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.loki import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
