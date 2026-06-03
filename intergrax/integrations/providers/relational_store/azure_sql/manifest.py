# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``azure_sql`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="azure_sql",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_AZURE_SQL',
    description='azure_sql integration (Phase M.6 P2/P3)',
)
