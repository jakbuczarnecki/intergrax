# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "OPENTELEMETRY_COLLECTOR_OBSERVABILITY_PROVIDER_ID",
    "OpenTelemetryCollectorObservabilityIntegration",
    "OpenTelemetryCollectorObservabilityIntegrationConfig",
    "OpenTelemetryCollectorObservabilityTransport",
    "create_opentelemetry_collector_observability_backend",
    "create_opentelemetry_collector_observability_integration",
    "register_opentelemetry_collector_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_opentelemetry_collector_observability_backend",
        "create_opentelemetry_collector_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "OPENTELEMETRY_COLLECTOR_OBSERVABILITY_PROVIDER_ID",
        "OpenTelemetryCollectorObservabilityIntegration",
        "OpenTelemetryCollectorObservabilityIntegrationConfig",
        "OpenTelemetryCollectorObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_opentelemetry_collector_integration":
        from intergrax.integrations.providers.observability_backend.opentelemetry_collector.register import (
            register_opentelemetry_collector_integration,
        )

        return register_opentelemetry_collector_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.opentelemetry_collector import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.opentelemetry_collector import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
