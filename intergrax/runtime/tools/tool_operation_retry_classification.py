# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

from __future__ import annotations

from enum import Enum

from intergrax.tools.core.contracts import SideEffectRetrySafety, ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from pydantic import BaseModel


class ToolOperationRetryClass(str, Enum):
    """
    Sanctioned operation safety classification for retry/idempotency (TOOL-LIFE-X §3.0.5).

    Derived only from typed ``ToolContract`` fields and idempotency key presence —
    never from HTTP method, tool name, or provider heuristics.
    """

    READ_ONLY = "read_only"
    NON_IDEMPOTENT_WRITE = "non_idempotent_write"
    IDEMPOTENT_WRITE = "idempotent_write"
    EXTERNALLY_DEDUPLICATED = "externally_deduplicated"
    EXPLICITLY_RETRY_SAFE_WRITE = "explicitly_retry_safe_write"


def classify_tool_operation_retry_class(
    contract: ToolContract,
    request: ToolExecutionRequest[BaseModel],
) -> ToolOperationRetryClass:
    if not contract.side_effects:
        return ToolOperationRetryClass.READ_ONLY
    safety = contract.side_effect_retry_safety
    if safety is SideEffectRetrySafety.NOT_RETRY_SAFE:
        return ToolOperationRetryClass.NON_IDEMPOTENT_WRITE
    if safety is SideEffectRetrySafety.IDEMPOTENT:
        return ToolOperationRetryClass.IDEMPOTENT_WRITE
    if request.idempotency_key is not None:
        return ToolOperationRetryClass.EXTERNALLY_DEDUPLICATED
    return ToolOperationRetryClass.EXPLICITLY_RETRY_SAFE_WRITE


def effective_max_retry_attempts(
    contract: ToolContract,
    request: ToolExecutionRequest[BaseModel],
) -> int:
    """Bounded automatic retry attempts for one logical tool invocation."""
    policy_attempts = contract.retry_policy.max_attempts
    operation_class = classify_tool_operation_retry_class(contract, request)
    if operation_class is ToolOperationRetryClass.READ_ONLY:
        return policy_attempts
    if operation_class is ToolOperationRetryClass.NON_IDEMPOTENT_WRITE:
        return 1
    if operation_class in (
        ToolOperationRetryClass.IDEMPOTENT_WRITE,
        ToolOperationRetryClass.EXTERNALLY_DEDUPLICATED,
    ):
        if request.idempotency_key is None:
            return 1
        return policy_attempts
    return policy_attempts
