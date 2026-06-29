# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Prometheus observability integration (Phase M.6)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.observability_backend.prometheus.config import (
    ENV_PROMETHEUS_BASE_URL,
    ENV_PROMETHEUS_BEARER_TOKEN,
    PrometheusIntegrationConfig,
)

__all__ = [

    "PROMETHEUS_OBSERVABILITY_PROVIDER_ID",
    "PrometheusObservabilityIntegration",
    "PrometheusObservabilityIntegrationConfig",
    "PrometheusObservabilityTransport",

    "ENV_PROMETHEUS_BASE_URL",
    "ENV_PROMETHEUS_BEARER_TOKEN",
    "PrometheusIntegrationBundle",
    "PrometheusIntegrationConfig",
    "PrometheusObservabilityBackend",
    "create_prometheus_integration",
    "create_prometheus_observability_backend",
    "create_prometheus_observability_integration",
    "register_prometheus_integration",
    "resolve_prometheus_config",
]


_INTEGRATION_EXPORTS = frozenset(
    {
        "PROMETHEUS_OBSERVABILITY_PROVIDER_ID",
        "PrometheusObservabilityIntegration",
        "PrometheusObservabilityIntegrationConfig",
        "PrometheusObservabilityTransport",
    }
)

_LAZY_EXPORTS = frozenset(
    {
        "PrometheusIntegrationBundle",
        "PrometheusObservabilityBackend",
        "create_prometheus_integration",
        "create_prometheus_observability_backend",
        "create_prometheus_observability_integration",
        "register_prometheus_integration",
        "resolve_prometheus_config",
    }
)


def __getattr__(name: str):
    if name == "register_prometheus_integration":
        from intergrax.integrations.providers.observability_backend.prometheus.register import register_prometheus_integration

        return register_prometheus_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.observability_backend.prometheus import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "PrometheusObservabilityBackend":
        from intergrax.integrations.providers.observability_backend.prometheus.adapter import _PrometheusObservabilityBackend

        return PrometheusObservabilityBackend

    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.prometheus import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
