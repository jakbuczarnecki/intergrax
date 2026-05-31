# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register arize."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.arize.bundle import create_arize_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_arize_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.ARIZE.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_arize_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_ARIZE",
            description="arize integration (Phase M.8 harness)",
        ),
        override=override,
    )
