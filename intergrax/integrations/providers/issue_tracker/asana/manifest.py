# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``asana`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="asana",
    categories=(IntegrationCategory.ISSUE_TRACKER,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_ASANA',
    description='asana integration (Phase M.6 P4)',
)
