# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``github_actions`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="github_actions",
    categories=(IntegrationCategory.CI_CD,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_GITHUB_ACTIONS',
    description='github_actions integration (Phase M.6 P4)',
)
