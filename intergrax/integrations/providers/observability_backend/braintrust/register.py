# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register braintrust."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.braintrust.bundle import create_braintrust_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_braintrust_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.BRAINTRUST.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_braintrust_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_BRAINTRUST",
            description="braintrust integration (Phase M.8 harness)",
        ),
        override=override,
    )
