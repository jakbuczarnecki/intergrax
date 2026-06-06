# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``falkordb`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="falkordb",
    categories=(IntegrationCategory.GRAPH_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_FALKORDB',
    description='falkordb integration (Phase M.6 P4)',
)
