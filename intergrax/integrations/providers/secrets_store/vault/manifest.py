# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``vault`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="vault",
    categories=(IntegrationCategory.SECRETS_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_VAULT',
    description='vault integration (Phase M.7)',
)
