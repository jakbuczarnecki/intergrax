# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "INFLUXDB_OBSERVABILITY_PROVIDER_ID",
    "InfluxdbObservabilityIntegration",
    "InfluxdbObservabilityIntegrationConfig",
    "InfluxdbObservabilityTransport",
    "create_influxdb_observability_backend",
    "create_influxdb_observability_integration",
    "register_influxdb_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_influxdb_observability_backend",
        "create_influxdb_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "INFLUXDB_OBSERVABILITY_PROVIDER_ID",
        "InfluxdbObservabilityIntegration",
        "InfluxdbObservabilityIntegrationConfig",
        "InfluxdbObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_influxdb_integration":
        from intergrax.integrations.providers.observability_backend.influxdb.register import (
            register_influxdb_integration,
        )

        return register_influxdb_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.influxdb import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.influxdb import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
