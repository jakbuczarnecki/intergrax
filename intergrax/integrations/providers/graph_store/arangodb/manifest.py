# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="arangodb",
    categories=(IntegrationCategory.GRAPH_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_ARANGODB",
    description="ArangoDB graph store — AQL HTTP bridge (H-INT-GRAPH-3)",
)
