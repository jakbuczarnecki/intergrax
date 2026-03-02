# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Queueing layer composition root.

This module is responsible for wiring concrete task queue providers
into the TaskQueueProviderRegistry.

It is the only place where registry and concrete providers are coupled.
"""

from intergrax.queueing.providers.kafka.kafka_task_queue import KafkaTaskQueue
from intergrax.queueing.providers.rabbitmq.rabbitmq_task_queue import RabbitMQTaskQueue
from intergrax.queueing.registry import TaskQueueProviderRegistry


def bootstrap_default_providers(
    registry: TaskQueueProviderRegistry,
) -> None:
    """
    Register default task queue providers.

    This function must be called during application composition phase.
    """
    registry.register("kafka", KafkaTaskQueue)
    registry.register("rabbitmq", RabbitMQTaskQueue)