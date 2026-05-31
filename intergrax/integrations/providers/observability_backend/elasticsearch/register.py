# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Elasticsearch in the integration catalog (Phase M.6 P2)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.elasticsearch.bundle import create_elasticsearch_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_elasticsearch_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.ELASTICSEARCH.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_elasticsearch_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_ELASTICSEARCH",
            description=(
                "Elasticsearch log search (_search aggregations; query_string via ObservabilityBackend)"
            ),
        ),
        override=override,
    )
