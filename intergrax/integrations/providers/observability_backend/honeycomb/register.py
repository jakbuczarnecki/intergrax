# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register honeycomb."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.honeycomb.bundle import create_honeycomb_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_honeycomb_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.HONEYCOMB.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_honeycomb_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_HONEYCOMB",
            description="honeycomb integration (Phase M.8 harness)",
        ),
        override=override,
    )
