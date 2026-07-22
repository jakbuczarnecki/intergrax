# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``telegram`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="telegram",
    categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_TELEGRAM',
    description='telegram integration (Phase M.7 P7 — dual notification + interaction)',
)
