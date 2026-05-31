# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register clickhouse."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.clickhouse.bundle import create_clickhouse_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_clickhouse_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.CLICKHOUSE.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_clickhouse_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_CLICKHOUSE",
            description="clickhouse integration (Phase M.7)",
        ),
        override=override,
    )
