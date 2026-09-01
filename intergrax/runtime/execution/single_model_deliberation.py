# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Single Model → canonical INFERENCE execution seam (DS-DELIB-02).

Maps deliberation input to neutral ExecutionRequest without selecting
ExecutionStrategy or invoking providers directly.
"""

from __future__ import annotations

from typing import TypeVar

from intergrax.contracts.single_model_strategy import (
    SingleModelDeliberationInput,
    SingleModelInferenceConfiguration,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.execution.request import ExecutionRequest

T = TypeVar("T")


def single_model_inference_execution_request(
    deliberation_input: SingleModelDeliberationInput[T],
    *,
    inference: SingleModelInferenceConfiguration,
) -> ExecutionRequest[tuple[ChatMessage, ...], T]:
    """Build canonical inference ExecutionRequest for Single Model deliberation."""
    return ExecutionRequest(
        input=deliberation_input.messages,
        output_type=deliberation_input.output_type,
        inference_profile_id=inference.inference_profile_id,
    )
