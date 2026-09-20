# © Artur Czarnecki. All rights reserved.

"""Plane C vision inference on product worker pools (AUDIT-IDEAL-29.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.modality_production_resolver import resolve_live_vision_profile
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile
from intergrax.model_inference.registry.vision_provider import VisionProvider
from intergrax.contracts.modality_profile import (
    ModalityPlane,
    ModalityProfile,
    production_plane_c_modality_profile as canonical_production_plane_c_modality_profile,
)


@dataclass(frozen=True, slots=True)
class ModalityProductWorkerWiring:
    enabled: bool
    execution_mode: ModalityExecutionMode
    vision_provider: VisionProvider
    require_deterministic_cv: bool


def production_plane_c_modality_profile() -> ModalityProfile:
    """Product preset for deterministic Plane C vision tools."""
    return canonical_production_plane_c_modality_profile()


def resolve_modality_product_worker_wiring(
    env: ApplicationEnvironmentProfile,
) -> ModalityProductWorkerWiring:
    """Product hosts route heavy vision inference through worker pools."""
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    enabled = is_product and env.features.modality_worker_pool_enabled
    mode = (
        ModalityExecutionMode.THREAD_POOL
        if enabled
        else ModalityExecutionMode.IN_PROCESS
    )
    vision = resolve_live_vision_profile()
    return ModalityProductWorkerWiring(
        enabled=enabled,
        execution_mode=mode,
        vision_provider=vision.provider,
        require_deterministic_cv=bool(
            env.modality_profile and env.modality_profile.require_deterministic_cv
        ),
    )


def resolve_product_modality_execution_profile(
    env: ApplicationEnvironmentProfile,
) -> ModalityExecutionProfile:
    """Execution profile for product worker-pool modality routing."""
    wiring = resolve_modality_product_worker_wiring(env)
    return ModalityExecutionProfile(
        mode=wiring.execution_mode,
        max_workers=4,
        celery_task_always_eager=not wiring.enabled,
    )
