# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
RabbitMQ integration — single public entry for all RabbitMQ-backed Tier-0 facades.

Implementation classes live under ``intergrax.queueing.providers.rabbitmq``;
compose them only through this package (``opens.py`` is internal).
"""

from intergrax.integrations.providers.message_bus.rabbitmq.bundle import (
    RabbitMQIntegrationBundle,
    build_rabbitmq_transport,
    create_rabbitmq_integration,
    create_rabbitmq_message_bus,
    create_rabbitmq_worker,
)

__all__ = [
    "RabbitMQIntegrationBundle",
    "build_rabbitmq_transport",
    "create_rabbitmq_integration",
    "create_rabbitmq_message_bus",
    "create_rabbitmq_worker",
]
