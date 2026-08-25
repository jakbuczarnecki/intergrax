# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete RabbitMQ integration bundle — the single composition root for RabbitMQ in Intergrax.

All runtime wiring (message bus, producer, consumer, worker, transport bundle) MUST use
this module or ``profile.resolve(IntegrationCategory.MESSAGE_BUS)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.providers.message_bus.rabbitmq.config import RabbitMQIntegrationConfig
from intergrax.integrations.providers.message_bus.rabbitmq.opens import (
    open_rabbitmq_consumer,
    open_rabbitmq_producer,
    open_rabbitmq_task_queue,
    open_rabbitmq_worker,
)
from intergrax.queueing.contracts.message_consumer import MessageConsumer
from intergrax.queueing.contracts.message_producer import MessageProducer
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


@dataclass(frozen=True)
class RabbitMQIntegrationBundle:
    """RabbitMQ-backed message bus and transport components sharing one config."""

    config: RabbitMQIntegrationConfig
    kv_store: DistributedKVStore
    message_bus: MessageBus
    producer: MessageProducer
    consumer: MessageConsumer


def resolve_rabbitmq_config(**overrides: object) -> RabbitMQIntegrationConfig:
    return RabbitMQIntegrationConfig.from_env(**overrides)


def _require_kv_store(kv_store: Optional[DistributedKVStore]) -> DistributedKVStore:
    if kv_store is None:
        raise ValueError(
            "RabbitMQ integration requires ``kv_store`` (typically from redis integration)."
        )
    return kv_store


def create_rabbitmq_integration(
    *,
    kv_store: Optional[DistributedKVStore] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    virtual_host: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    queue: Optional[str] = None,
    producer: Optional[MessageProducer] = None,
    consumer: Optional[MessageConsumer] = None,
    **config_overrides: object,
) -> RabbitMQIntegrationBundle:
    """Single entry point for RabbitMQ — message bus + producer + consumer."""
    overrides: dict[str, object] = dict(config_overrides)
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if virtual_host is not None:
        overrides["virtual_host"] = virtual_host
    if username is not None:
        overrides["username"] = username
    if password is not None:
        overrides["password"] = password
    if queue is not None:
        overrides["queue"] = queue

    config = resolve_rabbitmq_config(**overrides)
    store = _require_kv_store(kv_store)
    resolved_producer = open_rabbitmq_producer(config, producer=producer)
    resolved_consumer = open_rabbitmq_consumer(
        config,
        queue=queue,
        consumer=consumer,
    )
    bus = open_rabbitmq_task_queue(
        config,
        kv_store=store,
        queue=queue,
        producer=resolved_producer,
    )

    return RabbitMQIntegrationBundle(
        config=config,
        kv_store=store,
        message_bus=bus,
        producer=resolved_producer,
        consumer=resolved_consumer,
    )


def create_rabbitmq_message_bus(
    *,
    kv_store: Optional[DistributedKVStore] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    virtual_host: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    queue: Optional[str] = None,
    producer: Optional[MessageProducer] = None,
    **config_overrides: object,
) -> MessageBus:
    """Catalog factory for ``"rabbitmq"`` / ``MESSAGE_BUS``."""
    overrides: dict[str, object] = dict(config_overrides)
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if virtual_host is not None:
        overrides["virtual_host"] = virtual_host
    if username is not None:
        overrides["username"] = username
    if password is not None:
        overrides["password"] = password
    if queue is not None:
        overrides["queue"] = queue

    config = resolve_rabbitmq_config(**overrides)
    store = _require_kv_store(kv_store)
    resolved_producer = open_rabbitmq_producer(config, producer=producer)
    return open_rabbitmq_task_queue(
        config,
        kv_store=store,
        queue=queue,
        producer=resolved_producer,
    )


def create_rabbitmq_worker(
    *,
    kv_store: DistributedKVStore,
    execution_registry: TaskExecutionRegistry,
    idempotency_store: Optional[IdempotencyStore] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    virtual_host: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    queue: Optional[str] = None,
    consumer: Optional[MessageConsumer] = None,
    poll_timeout_seconds: float = 1.0,
    causal_evidence_persistence: CausalEvidencePersistence,
    **config_overrides: object,
) -> object:
    overrides: dict[str, object] = dict(config_overrides)
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if virtual_host is not None:
        overrides["virtual_host"] = virtual_host
    if username is not None:
        overrides["username"] = username
    if password is not None:
        overrides["password"] = password
    if queue is not None:
        overrides["queue"] = queue

    config = resolve_rabbitmq_config(**overrides)
    return open_rabbitmq_worker(
        config,
        kv_store=_require_kv_store(kv_store),
        registry=execution_registry,
        idempotency_store=idempotency_store,
        queue=queue or config.queue,
        consumer=consumer,
        poll_timeout_seconds=poll_timeout_seconds,
        causal_evidence_persistence=causal_evidence_persistence,
    )


def build_rabbitmq_transport(
    *,
    kv_store: DistributedKVStore,
    execution_registry: TaskExecutionRegistry,
    idempotency_store: Optional[IdempotencyStore] = None,
    queue: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    virtual_host: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    producer: Optional[MessageProducer] = None,
    consumer: Optional[MessageConsumer] = None,
    causal_evidence_persistence: CausalEvidencePersistence,
    **config_overrides: object,
) -> object:
    """
    Build ``TransportBundle`` for RabbitMQ (replaces inline wiring in ``runtime.transport``).
    """
    from intergrax.runtime.transport.bundle import TransportBundle

    bundle = create_rabbitmq_integration(
        kv_store=kv_store,
        host=host,
        port=port,
        virtual_host=virtual_host,
        username=username,
        password=password,
        queue=queue,
        producer=producer,
        consumer=consumer,
        **config_overrides,
    )
    worker = open_rabbitmq_worker(
        bundle.config,
        kv_store=bundle.kv_store,
        registry=execution_registry,
        idempotency_store=idempotency_store,
        queue=queue or bundle.config.queue,
        consumer=bundle.consumer,
        causal_evidence_persistence=causal_evidence_persistence,
    )
    return TransportBundle(task_queue=bundle.message_bus, worker=worker)

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.message_bus.rabbitmq.integration import (
    RABBITMQ_MESSAGE_BUS_PROVIDER_ID,
    RabbitmqMessageBusIntegration,
    RabbitmqMessageBusIntegrationConfig,
    RabbitmqMessageBusClient,
)


def create_rabbitmq_message_bus_integration(
    *,
    client: RabbitmqMessageBusClient | None = None,
    enabled: bool = False,
) -> RabbitmqMessageBusIntegration:
    """
    Build a contract-based Rabbitmq message bus integration.

    The legacy facade (create_rabbitmq_integration) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Rabbitmq message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return RabbitmqMessageBusIntegration.from_client(client, enabled=enabled)
    return RabbitmqMessageBusIntegration.for_provider(
        provider_id=RABBITMQ_MESSAGE_BUS_PROVIDER_ID,
        display_name="Rabbitmq",
        config=RabbitmqMessageBusIntegrationConfig(enabled=enabled),
    )
