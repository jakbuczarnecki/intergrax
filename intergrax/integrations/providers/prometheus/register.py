# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Prometheus in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.prometheus.bundle import create_prometheus_observability_backend
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_prometheus_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.PROMETHEUS.value,
            categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
            factory=create_prometheus_observability_backend,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_PROMETHEUS",
            description="Prometheus HTTP query API (query_instant, query_range via REST v1)",
        ),
        override=override,
    )
