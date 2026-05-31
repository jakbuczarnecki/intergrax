# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register nats."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.message_bus.nats.bundle import create_nats_message_bus
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_nats_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.NATS.value,
            categories=(IntegrationCategory.MESSAGE_BUS,),
            factory=create_nats_message_bus,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_NATS",
            description="nats integration (Phase M.7)",
        ),
        override=override,
    )
