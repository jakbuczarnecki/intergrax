# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``neo4j`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="neo4j",
    categories=(IntegrationCategory.GRAPH_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_NEO4J',
    description='neo4j integration (Phase M.7)',
)
