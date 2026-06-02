from __future__ import annotations

from intergrax.applications._shared.modality_celery_wiring import resolve_modality_celery_bundle
from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile
from intergrax.model_inference.workers import celery_app as celery_module


def test_resolve_modality_celery_bundle_eager_registers_task() -> None:
    celery_module.reset_modality_celery_app()
    profile = ModalityExecutionProfile(
        mode=ModalityExecutionMode.CELERY,
        celery_task_always_eager=True,
    )
    bundle = resolve_modality_celery_bundle(profile)
    assert bundle is not None
    assert celery_module.MODALITY_CELERY_TASK_NAME in bundle.app.tasks
    celery_module.reset_modality_celery_app()


def test_resolve_modality_celery_bundle_without_broker_returns_none() -> None:
    celery_module.reset_modality_celery_app()
    profile = ModalityExecutionProfile(mode=ModalityExecutionMode.IN_PROCESS)
    assert resolve_modality_celery_bundle(profile) is None
