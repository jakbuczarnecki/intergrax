# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``github`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="github",
    categories=(IntegrationCategory.ISSUE_TRACKER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_GITHUB',
    description='github integration (Phase M.6 P2/P3)',
)
