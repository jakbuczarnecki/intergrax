# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register snowflake."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.relational_store.snowflake.bundle import create_snowflake_relational_store
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_snowflake_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SNOWFLAKE.value,
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            factory=create_snowflake_relational_store,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_SNOWFLAKE",
            description="snowflake integration (Phase M.7)",
        ),
        override=override,
    )
