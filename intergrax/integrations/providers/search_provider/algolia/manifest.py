# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``algolia`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="algolia",
    categories=(IntegrationCategory.SEARCH_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_ALGOLIA',
    description='algolia integration (Phase M.6 P6)',
)
