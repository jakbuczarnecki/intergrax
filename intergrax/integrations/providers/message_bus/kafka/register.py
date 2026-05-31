# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Kafka in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_message_bus
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_kafka_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.KAFKA.value,
            categories=(IntegrationCategory.MESSAGE_BUS,),
            factory=create_kafka_message_bus,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_KAFKA",
            description=(
                "Kafka message bus — task queue, producer, consumer, worker "
                "(via create_kafka_integration; requires kv_store)"
            ),
        ),
        override=override,
    )
