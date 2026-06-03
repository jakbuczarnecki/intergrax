# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``google_workspace`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="google_workspace",
    categories=(IntegrationCategory.COLLABORATION_SUITE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_GOOGLE_WORKSPACE',
    description='google_workspace integration (Phase M.6 P2/P3)',
)
