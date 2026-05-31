# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register SQLite in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.relational_store.sqlite.bundle import create_sqlite_relational_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_sqlite_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SQLITE.value,
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=create_sqlite_relational_store,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_SQLITE",
            description=(
                "SQLite — relational facade + trace, events, checkpoints, HITL, "
                "task memory, experiments, idempotency, session, org profile "
                "(via create_sqlite_integration)"
            ),
        ),
        override=override,
    )
