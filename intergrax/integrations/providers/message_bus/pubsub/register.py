# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register pubsub."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.message_bus.pubsub.bundle import create_pubsub_message_bus
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_pubsub_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.PUBSUB.value,
            categories=(IntegrationCategory.MESSAGE_BUS,),
            factory=create_pubsub_message_bus,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_PUBSUB",
            description="pubsub integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
