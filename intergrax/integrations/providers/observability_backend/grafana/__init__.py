# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "GRAFANA_OBSERVABILITY_PROVIDER_ID",
    "GrafanaObservabilityIntegration",
    "GrafanaObservabilityIntegrationConfig",
    "GrafanaObservabilityTransport",
    "create_grafana_observability_backend",
    "create_grafana_observability_integration",
    "register_grafana_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_grafana_observability_backend",
        "create_grafana_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "GRAFANA_OBSERVABILITY_PROVIDER_ID",
        "GrafanaObservabilityIntegration",
        "GrafanaObservabilityIntegrationConfig",
        "GrafanaObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_grafana_integration":
        from intergrax.integrations.providers.observability_backend.grafana.register import (
            register_grafana_integration,
        )

        return register_grafana_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.grafana import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.grafana import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
