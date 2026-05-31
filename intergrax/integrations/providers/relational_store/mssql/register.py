# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register mssql."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.relational_store.mssql.bundle import create_mssql_relational_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_mssql_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.MSSQL.value,
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=create_mssql_relational_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_MSSQL",
            description="mssql integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
