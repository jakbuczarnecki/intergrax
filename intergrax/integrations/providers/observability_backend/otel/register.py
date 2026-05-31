# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register otel."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.otel.bundle import create_otel_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_otel_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.OTEL.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_otel_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_OTEL",
            description="otel integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
