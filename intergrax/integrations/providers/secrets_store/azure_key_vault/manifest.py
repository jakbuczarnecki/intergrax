# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``azure_key_vault`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="azure_key_vault",
    categories=(IntegrationCategory.SECRETS_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_AZURE_KEY_VAULT',
    description='azure_key_vault integration (Phase M.6 P4)',
)
