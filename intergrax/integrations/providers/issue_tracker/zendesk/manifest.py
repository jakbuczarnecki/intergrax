# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``zendesk`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="zendesk",
    categories=(IntegrationCategory.ISSUE_TRACKER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_ZENDESK',
    description='zendesk integration (Phase M.6 P6)',
)
