# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``duckdb`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="duckdb",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_DUCKDB',
    description='duckdb integration (Phase M.6 P4)',
)
