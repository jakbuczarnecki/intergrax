# © Artur Czarnecki. All rights reserved.

"""Optional Celery message_bus integration for modality execution (Tier-3 hosts)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile

if TYPE_CHECKING:
    from intergrax.integrations.providers.message_bus.celery.bundle import CeleryIntegrationBundle

MODALITY_CELERY_BUNDLE_EXTRA_KEY = "modality_celery_bundle"


def _resolve_celery_broker_url() -> str:
    return (
        os.getenv("INTERGRAX_MODALITY_CELERY_BROKER_URL")
        or os.getenv("CELERY_BROKER_URL")
        or os.getenv("INTERGRAX_CELERY_BROKER_URL")
        or ""
    ).strip()


def resolve_modality_celery_bundle(
    profile: ModalityExecutionProfile,
) -> "CeleryIntegrationBundle | None":
    """
    Build a shared Celery integration bundle when modality execution targets Celery.

    Returns ``None`` when no broker is configured and eager mode is off.
    """
    broker = (profile.celery_broker_url or _resolve_celery_broker_url()).strip()
    if profile.mode != ModalityExecutionMode.CELERY and not broker and not profile.celery_task_always_eager:
        return None
    if not broker and not profile.celery_task_always_eager:
        return None
    from intergrax.integrations.providers.message_bus.celery.bundle import create_celery_integration
    from intergrax.model_inference.workers.celery_app import register_modality_celery_task

    bundle = create_celery_integration(
        broker_url=broker or "memory://",
        task_always_eager=profile.celery_task_always_eager,
    )
    register_modality_celery_task(bundle.app)
    return bundle
