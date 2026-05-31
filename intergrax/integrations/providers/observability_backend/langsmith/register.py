# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register langsmith."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.langsmith.bundle import create_langsmith_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_langsmith_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.LANGSMITH.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_langsmith_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_LANGSMITH",
            description="langsmith integration (Phase M.8 harness)",
        ),
        override=override,
    )
