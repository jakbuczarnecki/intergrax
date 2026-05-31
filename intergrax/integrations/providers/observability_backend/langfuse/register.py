# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register langfuse."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.langfuse.bundle import create_langfuse_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_langfuse_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.LANGFUSE.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_langfuse_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_LANGFUSE",
            description="langfuse integration (Phase M.7)",
        ),
        override=override,
    )
