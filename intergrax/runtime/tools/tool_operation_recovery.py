# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""TOOL-LIFE-X Batch C — stalled/abandoned disposition and recovery helpers."""

from __future__ import annotations

from enum import Enum

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationPhysicalState,
    ExternalOperationState,
)
from intergrax.contracts.tool_execution_outcome import ToolEffectCertainty
from intergrax.runtime.external_operations.external_operation_ownership import (
    ProcessLocalExternalOperationOwner,
    reconcile_orphaned_running_operations,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.tools.tool_operation_retry_classification import (
    ToolOperationRetryClass,
    classify_tool_operation_retry_class,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from pydantic import BaseModel


class ToolInvocationTerminalDisposition(str, Enum):
    """Sanctioned projection of roadmap terminals on the tool seam (no parallel SM)."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"
    STALLED = "stalled"


def reconcile_orphaned_tool_external_operations(
    store: ExternalOperationStateStore,
    *,
    active_owner: ProcessLocalExternalOperationOwner,
) -> tuple[ExternalOperationState, ...]:
    """Worker-loss reconciliation before a new tool invocation (W4-C + TOOL-LIFE-X TL8)."""
    return reconcile_orphaned_running_operations(store, active_owner=active_owner)


def is_stalled_running_external_operation(
    record: ExternalOperationState,
    *,
    parent_execution_deadline_exceeded: bool,
) -> bool:
    """RUNNING without progress past parent deadline → STALLED (TL7; no heartbeat layer)."""
    if record.physical_state is not ExternalOperationPhysicalState.RUNNING:
        return False
    return parent_execution_deadline_exceeded


def must_abandon_uncertain_outcome(
    *,
    effect_certainty: ToolEffectCertainty,
    contract: ToolContract,
    request: ToolExecutionRequest[BaseModel],
) -> bool:
    """UNKNOWN physical / uncertain effect forbids blind retry (TL5 + TL6)."""
    if effect_certainty is not ToolEffectCertainty.UNCERTAIN:
        return False
    operation_class = classify_tool_operation_retry_class(contract, request)
    return operation_class is ToolOperationRetryClass.NON_IDEMPOTENT_WRITE


def must_abandon_after_retry_exhaustion(
    *,
    effect_may_have_started: bool,
    contract: ToolContract,
    request: ToolExecutionRequest[BaseModel],
) -> bool:
    if not effect_may_have_started:
        return False
    if not contract.side_effects:
        return False
    operation_class = classify_tool_operation_retry_class(contract, request)
    return operation_class is not ToolOperationRetryClass.READ_ONLY


def runtime_error_for_disposition(
    disposition: ToolInvocationTerminalDisposition,
) -> RuntimeErrorCode:
    if disposition is ToolInvocationTerminalDisposition.ABANDONED:
        return RuntimeErrorCode.ABANDONED
    if disposition is ToolInvocationTerminalDisposition.STALLED:
        return RuntimeErrorCode.STALLED
    if disposition is ToolInvocationTerminalDisposition.TIMED_OUT:
        return RuntimeErrorCode.TIMEOUT
    if disposition is ToolInvocationTerminalDisposition.CANCELLED:
        return RuntimeErrorCode.RUNTIME_ERROR
    return RuntimeErrorCode.TOOL_ERROR


def disposition_message(disposition: ToolInvocationTerminalDisposition) -> str:
    return disposition.value


__all__ = [
    "ToolInvocationTerminalDisposition",
    "disposition_message",
    "is_stalled_running_external_operation",
    "must_abandon_after_retry_exhaustion",
    "must_abandon_uncertain_outcome",
    "reconcile_orphaned_tool_external_operations",
    "runtime_error_for_disposition",
]
