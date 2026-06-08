# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``motherduck`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="motherduck",
    categories=(IntegrationCategory.RELATIONAL_STORE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_MOTHERDUCK',
    description='motherduck integration (Phase M.7 P7)',
)
