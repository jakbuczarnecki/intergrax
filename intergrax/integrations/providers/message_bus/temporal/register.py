# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register temporal."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.message_bus.temporal.bundle import create_temporal_message_bus
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_temporal_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.TEMPORAL.value,
            categories=(IntegrationCategory.MESSAGE_BUS,),
            factory=create_temporal_message_bus,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_TEMPORAL",
            description="temporal integration (Phase M.7)",
        ),
        override=override,
    )
