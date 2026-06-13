# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="neptune",
    categories=(IntegrationCategory.GRAPH_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_NEPTUNE",
    description="Amazon Neptune graph store — OpenCypher HTTP bridge (H-INT-GRAPH-1)",
)
