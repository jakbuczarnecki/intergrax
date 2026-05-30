# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register RabbitMQ in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.rabbitmq.bundle import create_rabbitmq_message_bus
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_rabbitmq_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.RABBITMQ.value,
            categories=(IntegrationCategory.MESSAGE_BUS,),
            factory=create_rabbitmq_message_bus,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_RABBITMQ",
            description=(
                "RabbitMQ message bus — task queue + worker "
                "(via create_rabbitmq_integration / build_rabbitmq_transport; requires kv_store)"
            ),
        ),
        override=override,
    )
