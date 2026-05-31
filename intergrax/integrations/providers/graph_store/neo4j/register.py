# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register neo4j."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.graph_store.neo4j.bundle import create_neo4j_graph_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_neo4j_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.NEO4J.value,
            categories=(IntegrationCategory.GRAPH_STORE,),
            factory=create_neo4j_graph_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_NEO4J",
            description="neo4j integration (Phase M.7)",
        ),
        override=override,
    )
