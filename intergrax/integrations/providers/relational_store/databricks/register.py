# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Databricks in the integration catalog (Phase M.6 P2)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.relational_store.databricks.bundle import create_databricks_relational_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_databricks_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.DATABRICKS.value,
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=create_databricks_relational_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_DATABRICKS",
            description=(
                "Databricks SQL Warehouse relational store (via create_databricks_integration); "
                "analytics / lakehouse reporting — domain stores remain SQLite-first"
            ),
        ),
        override=override,
    )
