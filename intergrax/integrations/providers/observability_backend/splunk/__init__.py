# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "SPLUNK_OBSERVABILITY_PROVIDER_ID",
    "SplunkObservabilityIntegration",
    "SplunkObservabilityIntegrationConfig",
    "SplunkObservabilityTransport",
    "create_splunk_observability_backend",
    "create_splunk_observability_integration",
    "register_splunk_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_splunk_observability_backend",
        "create_splunk_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "SPLUNK_OBSERVABILITY_PROVIDER_ID",
        "SplunkObservabilityIntegration",
        "SplunkObservabilityIntegrationConfig",
        "SplunkObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_splunk_integration":
        from intergrax.integrations.providers.observability_backend.splunk.register import (
            register_splunk_integration,
        )

        return register_splunk_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.splunk import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.splunk import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
