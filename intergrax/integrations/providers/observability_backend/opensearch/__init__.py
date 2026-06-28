# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "OPENSEARCH_OBSERVABILITY_PROVIDER_ID",
    "OpensearchObservabilityIntegration",
    "OpensearchObservabilityIntegrationConfig",
    "OpensearchObservabilityTransport",
    "create_opensearch_observability_backend",
    "create_opensearch_observability_integration",
    "register_opensearch_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_opensearch_observability_backend",
        "create_opensearch_observability_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "OPENSEARCH_OBSERVABILITY_PROVIDER_ID",
        "OpensearchObservabilityIntegration",
        "OpensearchObservabilityIntegrationConfig",
        "OpensearchObservabilityTransport",
    }
)


def __getattr__(name: str):
    if name == "register_opensearch_integration":
        from intergrax.integrations.providers.observability_backend.opensearch.register import (
            register_opensearch_integration,
        )

        return register_opensearch_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.observability_backend.opensearch import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.opensearch import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
