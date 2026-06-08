# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``arxiv`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="arxiv",
    categories=(IntegrationCategory.SEARCH_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_ARXIV',
    description='arxiv integration (Phase M.7 P7)',
)
