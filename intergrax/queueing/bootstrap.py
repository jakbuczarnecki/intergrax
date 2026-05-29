# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Queueing layer composition root.

Kafka/RabbitMQ class registration for ``TaskQueueProviderRegistry``.
Wiring of live instances: ``integrations.providers.kafka`` / ``runtime.transport.bootstrap``.
"""

from intergrax.queueing.providers.kafka.kafka_task_queue import KafkaTaskQueue
from intergrax.queueing.providers.rabbitmq.rabbitmq_task_queue import RabbitMQTaskQueue
from intergrax.queueing.registry import TaskQueueProviderRegistry


def bootstrap_default_providers(
    registry: TaskQueueProviderRegistry,
) -> None:
    """
    Register default task queue provider classes.

    Instance composition uses ``integrations.providers.kafka.create_kafka_integration``.
    """
    registry.register("kafka", KafkaTaskQueue)
    registry.register("rabbitmq", RabbitMQTaskQueue)
