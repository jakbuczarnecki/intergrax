# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register supabase."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.relational_store.supabase.bundle import create_supabase_relational_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_supabase_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SUPABASE.value,
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=create_supabase_relational_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_SUPABASE",
            description="supabase integration (Phase M.7)",
        ),
        override=override,
    )
