# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register MySQL in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.mysql.bundle import create_mysql_relational_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_mysql_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.MYSQL.value,
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=create_mysql_relational_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_MYSQL",
            description=(
                "MySQL relational store (via create_mysql_integration); "
                "domain stores remain SQLite-first until dedicated MySQL backends ship"
            ),
        ),
        override=override,
    )
