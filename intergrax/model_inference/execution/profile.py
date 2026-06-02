# © Artur Czarnecki. All rights reserved.

"""Execution routing profile for Plane C inference (Phase W-ML workers)."""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ModalityExecutionMode(str, Enum):
    """Where heavy vision/ML jobs run."""

    IN_PROCESS = "in_process"
    THREAD_POOL = "thread_pool"
    CELERY = "celery"


def _default_heavy_slugs() -> frozenset[str]:
    return frozenset({"yolo_ultralytics", "vision_serving", "huggingface_inference"})


class ModalityExecutionProfile(BaseModel):
    """
    Controls offload of GPU-heavy vision adapters to a background worker pool.

    ``thread_pool`` uses a bounded ``ThreadPoolExecutor`` (harness default for YOLO).
    ``celery`` dispatches heavy slugs via ``intergrax.modality.run_job`` when a broker URL is set.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    mode: ModalityExecutionMode = ModalityExecutionMode.IN_PROCESS
    max_workers: int = Field(default=4, ge=1, le=32)
    heavy_adapter_slugs: frozenset[str] = Field(default_factory=_default_heavy_slugs)
    celery_broker_url: str | None = None
    celery_task_always_eager: bool = False
    celery_result_timeout_s: int = Field(default=120, ge=1, le=3600)


def modality_execution_profile_from_env(*, prefix: str = "INTERGRAX_MODALITY") -> ModalityExecutionProfile:
    raw = (os.getenv(f"{prefix}_EXECUTION") or ModalityExecutionMode.IN_PROCESS.value).strip().lower()
    workers_raw = os.getenv(f"{prefix}_EXECUTION_WORKERS", "4").strip()
    max_workers = max(1, min(32, int(workers_raw)))
    broker = (
        os.getenv(f"{prefix}_CELERY_BROKER_URL")
        or os.getenv("CELERY_BROKER_URL")
        or os.getenv("INTERGRAX_CELERY_BROKER_URL")
        or None
    )
    eager = (os.getenv(f"{prefix}_CELERY_EAGER", "") or "").strip().lower() in {"1", "true", "yes"}
    timeout_raw = os.getenv(f"{prefix}_CELERY_TIMEOUT_S", "120").strip()
    return ModalityExecutionProfile(
        mode=ModalityExecutionMode(raw),
        max_workers=max_workers,
        celery_broker_url=broker,
        celery_task_always_eager=eager,
        celery_result_timeout_s=max(1, int(timeout_raw)),
    )
