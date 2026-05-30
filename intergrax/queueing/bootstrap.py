# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Queueing layer composition root.

Kafka and RabbitMQ MUST be composed via ``integrations.providers.kafka`` /
``integrations.providers.rabbitmq`` (or ``runtime.transport.bootstrap.build_transport``).
Celery via ``integrations.providers.celery``.

``TaskQueueProviderRegistry`` remains for optional extension; default brokers are not
registered here.
"""

from intergrax.queueing.registry import TaskQueueProviderRegistry


def bootstrap_default_providers(
    registry: TaskQueueProviderRegistry,
) -> None:
    """No-op — broker backends are registered in the Integration Library (Phase M.4)."""
    _ = registry
