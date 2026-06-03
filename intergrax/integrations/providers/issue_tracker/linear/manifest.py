# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``linear`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="linear",
    categories=(IntegrationCategory.ISSUE_TRACKER,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_LINEAR',
    description='linear integration (Phase M.6 P2/P3)',
)
