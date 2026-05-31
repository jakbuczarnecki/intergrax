# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register opensearch."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.opensearch.bundle import create_opensearch_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_opensearch_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.OPENSEARCH.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_opensearch_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_OPENSEARCH",
            description="opensearch integration (Phase M.8 harness)",
        ),
        override=override,
    )
