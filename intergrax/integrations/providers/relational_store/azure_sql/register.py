# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register azure_sql."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.relational_store.azure_sql.bundle import create_azure_sql_relational_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_azure_sql_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.AZURE_SQL.value,
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=create_azure_sql_relational_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_AZURE_SQL",
            description="azure_sql integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
