# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``cloudflare_r2`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="cloudflare_r2",
    categories=(IntegrationCategory.OBJECT_STORAGE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_CLOUDFLARE_R2',
    description='cloudflare_r2 integration (Phase M.6 P4)',
)
