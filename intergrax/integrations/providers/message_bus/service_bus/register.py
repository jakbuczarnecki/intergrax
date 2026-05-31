# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register service_bus."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.message_bus.service_bus.bundle import create_service_bus_message_bus
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_service_bus_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SERVICE_BUS.value,
            categories=(IntegrationCategory.MESSAGE_BUS,),
            factory=create_service_bus_message_bus,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_SERVICE_BUS",
            description="service_bus integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
