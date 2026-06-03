# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``ms365_graph`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="ms365_graph",
    categories=(IntegrationCategory.COLLABORATION_SUITE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_MS365',
    description='Microsoft 365 Graph (mail, calendar, directory via client credentials)',
)
