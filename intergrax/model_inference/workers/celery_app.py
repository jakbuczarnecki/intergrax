# © Artur Czarnecki. All rights reserved.

"""Celery application binding for modality inference tasks."""

from __future__ import annotations

import os
from typing import Optional

from celery import Celery

from intergrax.model_inference.workers.task_runner import run_modality_job

MODALITY_CELERY_TASK_NAME = "intergrax.modality.run_job"
_APP: Celery | None = None
_REGISTERED_APP: Celery | None = None


def register_modality_celery_task(app: Celery) -> None:
    """Register ``intergrax.modality.run_job`` on a host Celery app (idempotent)."""
    global _REGISTERED_APP
    if MODALITY_CELERY_TASK_NAME in app.tasks:
        _REGISTERED_APP = app
        return

    @app.task(name=MODALITY_CELERY_TASK_NAME)
    def modality_run_job(payload_json: str) -> str:
        return run_modality_job(payload_json)

    _REGISTERED_APP = app


def resolve_celery_broker_url() -> str:
    return (
        os.getenv("INTERGRAX_MODALITY_CELERY_BROKER_URL")
        or os.getenv("CELERY_BROKER_URL")
        or os.getenv("INTERGRAX_CELERY_BROKER_URL")
        or ""
    ).strip()


def reset_modality_celery_app() -> None:
    """Clear cached Celery app (tests and harness reconfiguration)."""
    global _APP, _REGISTERED_APP
    _APP = None
    _REGISTERED_APP = None


def get_modality_celery_app(*, broker_url: Optional[str] = None, task_always_eager: bool = False) -> Celery | None:
    global _APP
    if _REGISTERED_APP is not None:
        return _REGISTERED_APP
    resolved_broker = (broker_url if broker_url is not None else resolve_celery_broker_url()).strip()
    if not resolved_broker and not task_always_eager:
        return None
    if _APP is None:
        app = Celery("intergrax_modality", broker=resolved_broker or "memory://")
        app.conf.task_always_eager = task_always_eager
        app.conf.task_eager_propagates = True
        register_modality_celery_task(app)
        _APP = app
    return _APP
