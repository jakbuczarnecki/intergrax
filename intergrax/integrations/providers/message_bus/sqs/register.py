# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register sqs."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.message_bus.sqs.bundle import create_sqs_message_bus
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_sqs_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SQS.value,
            categories=(IntegrationCategory.MESSAGE_BUS,),
            factory=create_sqs_message_bus,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_SQS",
            description="sqs integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
