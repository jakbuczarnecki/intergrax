# © Artur Czarnecki. All rights reserved.

"""Modality inference execution plane (in-process and worker-pool routing)."""

from intergrax.model_inference.execution.executor import ModalityInferenceExecutor
from intergrax.model_inference.execution.factory import (
    MODALITY_EXECUTION_PROFILE_EXTRA_KEY,
    MODALITY_EXECUTOR_EXTRA_KEY,
    build_modality_inference_executor,
    modality_execution_profile_from_env,
)
from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile

__all__ = [
    "MODALITY_EXECUTION_PROFILE_EXTRA_KEY",
    "MODALITY_EXECUTOR_EXTRA_KEY",
    "ModalityExecutionMode",
    "ModalityExecutionProfile",
    "ModalityInferenceExecutor",
    "build_modality_inference_executor",
    "modality_execution_profile_from_env",
]
