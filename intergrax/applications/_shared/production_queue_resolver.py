# © Artur Czarnecki. All rights reserved.

"""Production queue backend resolver (AUDIT-IDEAL-9.1)."""

from __future__ import annotations

import os
from enum import Enum


class ProductionQueueBackend(str, Enum):
    INLINE = "inline"
    CELERY = "celery"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"


def resolve_production_queue_backend(
    *,
    env_value: str | None = None,
) -> ProductionQueueBackend:
    """Resolve queue backend from ``INTERGRAX_QUEUE_BACKEND`` (default: inline)."""
    raw = (env_value if env_value is not None else os.getenv("INTERGRAX_QUEUE_BACKEND", "")).strip().lower()
    if raw in ("celery",):
        return ProductionQueueBackend.CELERY
    if raw in ("rabbitmq", "amqp"):
        return ProductionQueueBackend.RABBITMQ
    if raw in ("kafka",):
        return ProductionQueueBackend.KAFKA
    return ProductionQueueBackend.INLINE


def production_queue_requires_worker(backend: ProductionQueueBackend) -> bool:
    return backend is not ProductionQueueBackend.INLINE
