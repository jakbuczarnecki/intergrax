# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Complete Celery integration bundle — the single composition root for Celery in Intergrax.

All runtime wiring (message bus, worker app, Nexus Celery stack) MUST use this module
or ``profile.resolve(IntegrationCategory.MESSAGE_BUS)`` with ``"celery"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.distributed.contracts.rate_limiter import DistributedRateLimiter
from intergrax.integrations.contracts.message_bus import MessageBus
from intergrax.integrations.providers.message_bus.celery.config import CeleryIntegrationConfig
from intergrax.integrations.providers.message_bus.celery.opens import open_celery_app, open_celery_task_queue
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.queueing.worker.rate_limit_event import RateLimitEvent
from intergrax.queueing.worker.retry_event import RetryEvent


@dataclass(frozen=True)
class CeleryIntegrationBundle:
    """Celery message bus + application instance sharing one config."""

    config: CeleryIntegrationConfig
    app: object
    message_bus: MessageBus


def resolve_celery_config(**overrides: object) -> CeleryIntegrationConfig:
    return CeleryIntegrationConfig.from_env(**overrides)


def create_celery_integration(
    *,
    app: Optional[object] = None,
    broker_url: Optional[str] = None,
    backend_url: Optional[str] = None,
    app_name: Optional[str] = None,
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
    app: Optional[object] = None,
    broker_url: Optional[str] = None,
    backend_url: Optional[str] = None,
    app_name: Optional[str] = None,
    task_always_eager: bool = False,
    **config_overrides: object,
) -> MessageBus:
    """Catalog factory for ``"celery"`` / ``MESSAGE_BUS``."""
    return create_celery_integration(
        app=app,
        broker_url=broker_url,
        backend_url=backend_url,
        app_name=app_name,
        task_always_eager=task_always_eager,
        **config_overrides,
    ).message_bus


def create_celery_worker_app(
    *,
    registry: TaskExecutionRegistry,
    broker_url: Optional[str] = None,
    backend_url: Optional[str] = None,
    app_name: Optional[str] = None,
    idempotency_store: Optional[IdempotencyStore] = None,
    retry_policy: Optional[RetryPolicy] = None,
    lock_ttl_seconds: Optional[int] = None,
    completed_ttl_seconds: Optional[int] = None,
    rate_limiter: Optional[DistributedRateLimiter] = None,
    rate_limit_config: Optional[Callable[[str], Tuple[int, float]]] = None,
    on_rate_limited: Optional[Callable[[RateLimitEvent], None]] = None,
    on_retry_scheduled: Optional[Callable[[RetryEvent], None]] = None,
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
    )


def create_nexus_celery_worker_app(**kwargs: object) -> object:
    """Lab/production Nexus stack — delegates to ``runtime.task.worker_bootstrap``."""
    from intergrax.runtime.task.worker_bootstrap import create_nexus_celery_worker_app as _create

    return _create(**kwargs)
