# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "NEWRELIC_OBSERVABILITY_PROVIDER_ID",
    "NewrelicObservabilityIntegration",
    "NewrelicObservabilityIntegrationConfig",
    "NewrelicObservabilityTransport",
    "create_newrelic_observability_backend",
    "create_newrelic_observability_integration",
    "register_newrelic_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_newrelic_observability_backend",
        "create_newrelic_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "NEWRELIC_OBSERVABILITY_PROVIDER_ID",
        "NewrelicObservabilityIntegration",
        "NewrelicObservabilityIntegrationConfig",
        "NewrelicObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_newrelic_integration":
        from intergrax.integrations.providers.observability_backend.newrelic.register import (
            register_newrelic_integration,
        )

        return register_newrelic_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.newrelic import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.newrelic import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
