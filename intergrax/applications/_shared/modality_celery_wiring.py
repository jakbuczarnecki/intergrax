# © Artur Czarnecki. All rights reserved.

"""Optional Celery message_bus integration for modality execution (Tier-3 hosts)."""

from __future__ import annotations

from intergrax.integrations.providers.message_bus.celery.bundle import (
    CeleryIntegrationBundle,
    create_celery_integration,
)
from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile
from intergrax.model_inference.workers.celery_app import (
    register_modality_celery_task,
    resolve_celery_broker_url,
)

MODALITY_CELERY_BUNDLE_EXTRA_KEY = "modality_celery_bundle"


def resolve_modality_celery_bundle(
    profile: ModalityExecutionProfile,
) -> CeleryIntegrationBundle | None:
    """
    Build a shared Celery integration bundle when modality execution targets Celery.

    Returns ``None`` when no broker is configured and eager mode is off.
    """
    broker = (profile.celery_broker_url or resolve_celery_broker_url()).strip()
    if profile.mode != ModalityExecutionMode.CELERY and not broker and not profile.celery_task_always_eager:
        return None
    if not broker and not profile.celery_task_always_eager:
        return None
    bundle = create_celery_integration(
        broker_url=broker or "memory://",
        task_always_eager=profile.celery_task_always_eager,
    )
    register_modality_celery_task(bundle.app)
    return bundle
