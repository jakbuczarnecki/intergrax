# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``wikipedia`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="wikipedia",
    categories=(IntegrationCategory.WIKI_KNOWLEDGE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_WIKIPEDIA',
    description='wikipedia integration (Phase M.7 P7)',
)
