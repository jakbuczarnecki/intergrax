# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Elasticsearch observability integration (Phase M.6 P2)."""

from intergrax.utils.lazy_export import export_from_bundle
from intergrax.integrations.providers.observability_backend.elasticsearch.config import (
    ENV_ELASTICSEARCH_API_KEY,
    ENV_ELASTICSEARCH_INDEX,
    ENV_ELASTICSEARCH_URL,
    ElasticsearchIntegrationConfig,
)

__all__ = [

    "ELASTICSEARCH_OBSERVABILITY_PROVIDER_ID",
    "ElasticsearchObservabilityIntegration",
    "ElasticsearchObservabilityIntegrationConfig",
    "ElasticsearchObservabilityTransport",

    "ENV_ELASTICSEARCH_API_KEY",
    "ENV_ELASTICSEARCH_INDEX",
    "ENV_ELASTICSEARCH_URL",
    "ElasticsearchIntegrationBundle",
    "ElasticsearchIntegrationConfig",
    "ElasticsearchObservabilityBackend",
    "create_elasticsearch_integration",
    "create_elasticsearch_observability_backend",
    "create_elasticsearch_observability_integration",
    "register_elasticsearch_integration",
    "resolve_elasticsearch_config",
]


_INTEGRATION_EXPORTS = frozenset(
    {
        "ELASTICSEARCH_OBSERVABILITY_PROVIDER_ID",
        "ElasticsearchObservabilityIntegration",
        "ElasticsearchObservabilityIntegrationConfig",
        "ElasticsearchObservabilityTransport",
    }
)

_LAZY_EXPORTS = frozenset(
    {
        "ElasticsearchIntegrationBundle",
        "ElasticsearchObservabilityBackend",
        "create_elasticsearch_integration",
        "create_elasticsearch_observability_backend",
        "create_elasticsearch_observability_integration",
        "register_elasticsearch_integration",
        "resolve_elasticsearch_config",
    }
)


def __getattr__(name: str):
    if name == "register_elasticsearch_integration":
        from intergrax.integrations.providers.observability_backend.elasticsearch.register import register_elasticsearch_integration

        return register_elasticsearch_integration
    if name in _LAZY_EXPORTS:
        from intergrax.integrations.providers.observability_backend.elasticsearch import bundle as _bundle

        return export_from_bundle(_bundle, name, _LAZY_EXPORTS)
    if name == "ElasticsearchObservabilityBackend":
        from intergrax.integrations.providers.observability_backend.elasticsearch.integration import (
            ElasticsearchObservabilityIntegration,
        )

        return ElasticsearchObservabilityIntegration

    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.observability_backend.elasticsearch import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
