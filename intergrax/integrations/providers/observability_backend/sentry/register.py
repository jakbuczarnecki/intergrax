# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register sentry."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.observability_backend.sentry.bundle import create_sentry_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_sentry_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SENTRY.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_sentry_observability_backend,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_SENTRY",
            description="sentry integration (Phase M.7)",
        ),
        override=override,
    )
