# © Artur Czarnecki. All rights reserved.

"""Factory for modality inference executors."""

from __future__ import annotations

from intergrax.model_inference.execution.executor import ModalityInferenceExecutor
from intergrax.model_inference.execution.in_process_executor import InProcessModalityInferenceExecutor
from intergrax.model_inference.execution.profile import (
    ModalityExecutionMode,
    ModalityExecutionProfile,
    modality_execution_profile_from_env,
)
from intergrax.model_inference.execution.thread_pool_executor import ThreadPoolModalityInferenceExecutor

MODALITY_EXECUTOR_EXTRA_KEY = "modality_inference_executor"
MODALITY_EXECUTION_PROFILE_EXTRA_KEY = "modality_execution_profile"


def build_modality_inference_executor(
    profile: ModalityExecutionProfile | None = None,
) -> ModalityInferenceExecutor:
    resolved = profile or modality_execution_profile_from_env()
    if resolved.mode == ModalityExecutionMode.CELERY:
        from intergrax.model_inference.execution.celery_executor import CeleryModalityInferenceExecutor

        return CeleryModalityInferenceExecutor(profile=resolved)
    if resolved.mode == ModalityExecutionMode.THREAD_POOL:
        return ThreadPoolModalityInferenceExecutor(profile=resolved)
    return InProcessModalityInferenceExecutor()
