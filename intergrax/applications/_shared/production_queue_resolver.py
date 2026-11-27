# © Artur Czarnecki. All rights reserved.

"""Production queue backend resolver (AUDIT-IDEAL-9.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, StrEnum

_DEFAULT_CELERY_BACKEND_URL = "redis://localhost:6379/2"


class ProductionQueueBackend(str, Enum):  # noqa: UP042
    INLINE = "inline"
    CELERY = "celery"
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"


class UnsupportedQueueBackendReason(StrEnum):
    """Why a queue-worker composition was refused (T-IGX-043-B, DP-30)."""

    BROKER_WIRING_NOT_BUILT = "BROKER_WIRING_NOT_BUILT"
    EAGER_WORKER_FORBIDDEN_IN_PRODUCTION = "EAGER_WORKER_FORBIDDEN_IN_PRODUCTION"
    CELERY_BROKER_URL_REQUIRED = "CELERY_BROKER_URL_REQUIRED"
    REFUSED_IN_PROCESS_BROKER = "REFUSED_IN_PROCESS_BROKER"
    TASK_STORE_NOT_DURABLE = "TASK_STORE_NOT_DURABLE"


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


_REFUSED_BROKER_PREFIXES = ("memory://", "cache+memory://")


def broker_url_refused_in_strict_posture(broker_url: str) -> bool:
    normalized = broker_url.strip().lower()
    return not normalized or any(normalized.startswith(prefix) for prefix in _REFUSED_BROKER_PREFIXES)


@dataclass(frozen=True, slots=True)
class CeleryQueueEnvConfig:
    """Celery broker settings resolved from the environment (Tier-3 composition boundary)."""

    app_name: str
    broker_url: str
    backend_url: str


def resolve_celery_queue_env_config(
    *,
    app_name: str | None = None,
) -> CeleryQueueEnvConfig:
    """Resolve Celery queue wiring from env vars without importing provider packages."""
    broker_raw = os.environ.get("INTERGRAX_CELERY_BROKER_URL")
    if broker_raw is None:
        broker = ""
    else:
        broker = broker_raw.strip()
    backend = (
        os.environ.get("INTERGRAX_CELERY_BACKEND_URL", _DEFAULT_CELERY_BACKEND_URL).strip()
        or _DEFAULT_CELERY_BACKEND_URL
    )
    default_app_name = app_name or "tier3_host_task_worker"
    resolved_app_name = os.environ.get("INTERGRAX_CELERY_APP_NAME", default_app_name).strip() or default_app_name
    return CeleryQueueEnvConfig(
        app_name=resolved_app_name,
        broker_url=broker,
        backend_url=backend,
    )


def require_celery_broker_url_for_strict(*, broker_url: str) -> None:
    """Fail closed when strict/production queue wiring needs an explicit broker URL."""
    if not broker_url.strip():
        raise UnsupportedQueueBackendError(
            UnsupportedQueueBackendReason.CELERY_BROKER_URL_REQUIRED,
            "INTERGRAX_CELERY_BROKER_URL must be set for out-of-process queue execution",
        )
    if broker_url_refused_in_strict_posture(broker_url):
        raise UnsupportedQueueBackendError(
            UnsupportedQueueBackendReason.REFUSED_IN_PROCESS_BROKER,
            f"broker URL {broker_url!r} is refused in strict posture",
        )
