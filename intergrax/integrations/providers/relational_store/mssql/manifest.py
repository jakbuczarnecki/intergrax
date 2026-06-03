# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``mssql`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="mssql",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_MSSQL',
    description='mssql integration (Phase M.6 P2/P3)',
)
