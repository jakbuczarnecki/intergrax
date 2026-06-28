# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "BRAINTRUST_OBSERVABILITY_PROVIDER_ID",
    "BraintrustObservabilityIntegration",
    "BraintrustObservabilityIntegrationConfig",
    "BraintrustObservabilityTransport",
    "create_braintrust_observability_backend",
    "create_braintrust_observability_integration",
    "register_braintrust_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_braintrust_observability_backend",
        "create_braintrust_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "BRAINTRUST_OBSERVABILITY_PROVIDER_ID",
        "BraintrustObservabilityIntegration",
        "BraintrustObservabilityIntegrationConfig",
        "BraintrustObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_braintrust_integration":
        from intergrax.integrations.providers.observability_backend.braintrust.register import (
            register_braintrust_integration,
        )

        return register_braintrust_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.braintrust import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.braintrust import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
