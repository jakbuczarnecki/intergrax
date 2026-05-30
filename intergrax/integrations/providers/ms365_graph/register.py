# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register MS365 Graph in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.ms365_graph.bundle import create_ms365_graph_collaboration_suite
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_ms365_graph_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.MS365_GRAPH.value,
            categories=(IntegrationCategory.COLLABORATION_SUITE,),
            factory=create_ms365_graph_collaboration_suite,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_MS365",
            description=(
                "Microsoft 365 Graph (mail, calendar, directory via client credentials)"
            ),
        ),
        override=override,
    )
