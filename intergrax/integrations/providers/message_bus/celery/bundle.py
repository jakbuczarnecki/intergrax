# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Celery integration bundle — the single composition root for Celery in Intergrax.

All runtime wiring (message bus, worker app, Nexus Celery stack) MUST use this module
or ``profile.resolve(IntegrationCategory.MESSAGE_BUS)`` with ``"celery"``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.rate_limiter import DistributedRateLimiter
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.providers.message_bus.celery.config import CeleryIntegrationConfig
from intergrax.integrations.providers.message_bus.celery.integration import (
    CELERY_MESSAGE_BUS_PROVIDER_ID,
    CeleryMessageBusClient,
    CeleryMessageBusIntegration,
    CeleryMessageBusIntegrationConfig,
)
from intergrax.integrations.providers.message_bus.celery.opens import open_celery_app, open_celery_task_queue
from intergrax.queueing.worker.rate_limit_event import RateLimitEvent
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_event import RetryEvent
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)


@dataclass(frozen=True)
class CeleryIntegrationBundle:
    """Celery message bus + application instance sharing one config."""

    config: CeleryIntegrationConfig
    app: object
    message_bus: MessageBus


def resolve_celery_config(**overrides: object) -> CeleryIntegrationConfig:
    return CeleryIntegrationConfig.from_env(**overrides)


def create_celery_message_bus_integration(
    *,
    client: CeleryMessageBusClient | None = None,
    enabled: bool = False,
) -> CeleryMessageBusIntegration:
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Celery message bus integration requires an injected client when enabled=True",
        )
    if client is not None:
        return CeleryMessageBusIntegration.from_client(client, enabled=enabled)
    return CeleryMessageBusIntegration.for_provider(
        provider_id=CELERY_MESSAGE_BUS_PROVIDER_ID,
        display_name="Celery",
        config=CeleryMessageBusIntegrationConfig(enabled=enabled),
    )


def create_celery_integration(
    *,
    app: object | None = None,
    broker_url: str | None = None,
    backend_url: str | None = None,
    app_name: str | None = None,
    task_always_eager: bool = False,
    **config_overrides: object,
) -> CeleryIntegrationBundle:
    """Single entry point for Celery — config, app, and message bus."""
    overrides: dict[str, object] = dict(config_overrides)
    if broker_url is not None:
        overrides["broker_url"] = broker_url
    if backend_url is not None:
        overrides["backend_url"] = backend_url
    if app_name is not None:
        overrides["app_name"] = app_name

    config = resolve_celery_config(**overrides)
    resolved_app = open_celery_app(config, app=app, task_always_eager=task_always_eager)
    bus = open_celery_task_queue(config, app=resolved_app, task_always_eager=task_always_eager)

    return CeleryIntegrationBundle(
        config=config,
        app=resolved_app,
        message_bus=bus,
    )


def create_celery_message_bus(
    *,
    app: object | None = None,
    broker_url: str | None = None,
    backend_url: str | None = None,
    app_name: str | None = None,
    task_always_eager: bool = False,
    **config_overrides: object,
) -> CeleryMessageBusIntegration:
    """Catalog factory for ``"celery"`` / ``MESSAGE_BUS``."""
    if app is None and broker_url is None and backend_url is None and not task_always_eager:
        return create_celery_message_bus_integration(enabled=False)
    bus = create_celery_integration(
        app=app,
        broker_url=broker_url,
        backend_url=backend_url,
        app_name=app_name,
        task_always_eager=task_always_eager,
        **config_overrides,
    ).message_bus
    return CeleryMessageBusIntegration.from_client(bus, enabled=True)


def create_celery_worker_app(
    *,
    registry: TaskExecutionRegistry,
    broker_url: str | None = None,
    backend_url: str | None = None,
    app_name: str | None = None,
    idempotency_store: IdempotencyStore | None = None,
    retry_policy: RetryPolicy | None = None,
    lock_ttl_seconds: int | None = None,
    completed_ttl_seconds: int | None = None,
    rate_limiter: DistributedRateLimiter | None = None,
    rate_limit_config: Callable[[str], tuple[int, float]] | None = None,
    on_rate_limited: Callable[[RateLimitEvent], None] | None = None,
    on_retry_scheduled: Callable[[RetryEvent], None] | None = None,
    causal_evidence_persistence: CausalEvidencePersistence,
    **config_overrides: object,
) -> object:
    """Tier-0 worker app with dispatcher registered (``intergrax.execute``)."""
    from intergrax.queueing.worker_bootstrap import create_celery_worker_app as _create

    config = resolve_celery_config(
        **{
            **config_overrides,
            **({} if broker_url is None else {"broker_url": broker_url}),
            **({} if backend_url is None else {"backend_url": backend_url}),
            **({} if app_name is None else {"app_name": app_name}),
        }
    )
    return _create(
        app_name=config.app_name,
        broker_url=config.broker_url,
        backend_url=config.backend_url,
        registry=registry,
        idempotency_store=idempotency_store,
        retry_policy=retry_policy,
        lock_ttl_seconds=lock_ttl_seconds,
        completed_ttl_seconds=completed_ttl_seconds,
        rate_limiter=rate_limiter,
        rate_limit_config=rate_limit_config,
        on_rate_limited=on_rate_limited,
        on_retry_scheduled=on_retry_scheduled,
        causal_evidence_persistence=causal_evidence_persistence,
    )


def create_nexus_celery_worker_app(**kwargs: object) -> object:
    """Lab/production Nexus stack — delegates to ``runtime.task.worker_bootstrap``."""
    from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app as _create

    return _create(**kwargs)
