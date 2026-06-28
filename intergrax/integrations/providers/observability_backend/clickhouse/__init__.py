# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "CLICKHOUSE_OBSERVABILITY_PROVIDER_ID",
    "ClickhouseObservabilityIntegration",
    "ClickhouseObservabilityIntegrationConfig",
    "ClickhouseObservabilityTransport",
    "create_clickhouse_observability_backend",
    "create_clickhouse_observability_integration",
    "register_clickhouse_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_clickhouse_observability_backend",
        "create_clickhouse_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "CLICKHOUSE_OBSERVABILITY_PROVIDER_ID",
        "ClickhouseObservabilityIntegration",
        "ClickhouseObservabilityIntegrationConfig",
        "ClickhouseObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_clickhouse_integration":
        from intergrax.integrations.providers.observability_backend.clickhouse.register import (
            register_clickhouse_integration,
        )

        return register_clickhouse_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.clickhouse import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.clickhouse import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
