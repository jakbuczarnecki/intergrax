# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register vault."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.secrets_store.vault.bundle import create_vault_secrets_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_vault_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.VAULT.value,
            categories=(IntegrationCategory.SECRETS_STORE,),
            factory=create_vault_secrets_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_VAULT",
            description="vault integration (Phase M.7)",
        ),
        override=override,
    )
