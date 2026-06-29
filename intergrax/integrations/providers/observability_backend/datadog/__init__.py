# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "DATADOG_OBSERVABILITY_PROVIDER_ID",
    "DatadogObservabilityIntegration",
    "DatadogObservabilityIntegrationConfig",
    "DatadogObservabilityTransport",
    "create_datadog_observability_backend",
    "create_datadog_observability_integration",
    "register_datadog_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_datadog_observability_backend",
        "create_datadog_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "DATADOG_OBSERVABILITY_PROVIDER_ID",
        "DatadogObservabilityIntegration",
        "DatadogObservabilityIntegrationConfig",
        "DatadogObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_datadog_integration":
        from intergrax.integrations.providers.observability_backend.datadog.register import (
            register_datadog_integration,
        )

        return register_datadog_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.datadog import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.datadog import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
