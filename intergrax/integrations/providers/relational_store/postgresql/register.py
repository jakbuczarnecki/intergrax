# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register PostgreSQL in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.relational_store.postgresql.bundle import create_postgresql_relational_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_postgresql_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.POSTGRESQL.value,
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=create_postgresql_relational_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_POSTGRESQL",
            description=(
                "PostgreSQL relational store (via create_postgresql_integration); "
                "domain stores remain SQLite-first until dedicated Postgres backends ship"
            ),
        ),
        override=override,
    )
