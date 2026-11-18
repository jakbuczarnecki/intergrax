# © Artur Czarnecki. All rights reserved.

"""Production queue backend resolver (AUDIT-IDEAL-9.1)."""

from __future__ import annotations

import os
from enum import Enum, StrEnum


class ProductionQueueBackend(str, Enum):  # noqa: UP042
    INLINE = "inline"
    CELERY = "celery"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"


class UnsupportedQueueBackendReason(StrEnum):
    """Why a queue-worker composition was refused (T-IGX-043-B, DP-30)."""

    BROKER_WIRING_NOT_BUILT = "BROKER_WIRING_NOT_BUILT"
    EAGER_WORKER_FORBIDDEN_IN_PRODUCTION = "EAGER_WORKER_FORBIDDEN_IN_PRODUCTION"


class UnsupportedQueueBackendError(ValueError):
    """Raised for queue-worker compositions that are not built or not allowed in production."""

    def __init__(self, reason: UnsupportedQueueBackendReason, message: str) -> None:
        super().__init__(f"{reason.value}: {message}")
        self.reason = reason


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
