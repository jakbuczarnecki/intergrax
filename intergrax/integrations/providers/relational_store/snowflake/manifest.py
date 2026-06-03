# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``snowflake`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="snowflake",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_SNOWFLAKE',
    description='snowflake integration (Phase M.7)',
)
