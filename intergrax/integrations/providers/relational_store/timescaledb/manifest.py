# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``timescaledb`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="timescaledb",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_TIMESCALEDB',
    description='timescaledb integration (Phase M.6 P4)',
)
