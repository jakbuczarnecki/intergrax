# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``exa`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="exa",
    categories=(IntegrationCategory.SEARCH_PROVIDER,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_EXA',
    description='exa integration (Phase M.7)',
)
