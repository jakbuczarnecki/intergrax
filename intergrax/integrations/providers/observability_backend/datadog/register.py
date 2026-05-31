# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register datadog."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.datadog.bundle import create_datadog_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_datadog_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.DATADOG.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_datadog_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_DATADOG",
            description="datadog integration (Phase M.7)",
        ),
        override=override,
    )
