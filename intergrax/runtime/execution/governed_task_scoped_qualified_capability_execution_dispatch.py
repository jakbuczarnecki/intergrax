# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bind canonical governed Task for qualified capability dispatch scope (UCA-6C-R6-R5.8-R2-H1)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionAsyncDispatchPort,
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchPort,
    QualifiedCapabilityExecutionDispatchRequest,
    QualifiedCapabilityExecutionDispatchResult,
)
from intergrax.contracts.execution_identity import TaskId, validate_task_id
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task


class GovernedExecutionTaskLookupPort(Protocol):
    """Resolve live governed Task for execution correlation — not a second task store."""

    def resolve_task(self, task_id: TaskId) -> Task | None: ...


class ActiveTaskRegistryGovernedExecutionTaskLookup:
    """Process-local governed task lookup via canonical ActiveTaskRegistry."""

    def resolve_task(self, task_id: TaskId) -> Task | None:
        validated = validate_task_id(task_id)
        binding = ActiveTaskRegistry.peek_binding(validated)
        if binding is None:
            return None
        if binding.task_id != validated:
            return None
        return binding.task


class GovernedTaskScopedQualifiedCapabilityExecutionDispatchService(
    QualifiedCapabilityExecutionDispatchPort,
    QualifiedCapabilityExecutionAsyncDispatchPort,
):
    """Scope governed Task ContextVar to dispatch — matches TaskId on the request."""

    def __init__(
        self,
        *,
        inner: QualifiedCapabilityExecutionDispatchPort,
        task_lookup: GovernedExecutionTaskLookupPort,
    ) -> None:
        self._inner = inner
        self._task_lookup = task_lookup
        if not isinstance(inner, QualifiedCapabilityExecutionAsyncDispatchPort):
            raise TypeError(
                "inner dispatch must implement QualifiedCapabilityExecutionAsyncDispatchPort",
            )
        self._inner_async: QualifiedCapabilityExecutionAsyncDispatchPort = inner

    def dispatch(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult:
        task = self._task_lookup.resolve_task(request.task_id)
        if task is None:
            return QualifiedCapabilityExecutionDispatchResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                execution_request_id=request.execution_request_id,
                reason_detail="governed_execution_task_unavailable",
            )
        if task.task_id != request.task_id:
            return QualifiedCapabilityExecutionDispatchResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                execution_request_id=request.execution_request_id,
                reason_detail="governed_execution_task_id_mismatch",
            )
        token = bind_governed_execution_task(task)
        try:
            return self._inner.dispatch(request)
        finally:
            reset_governed_execution_task(token)

    async def dispatch_async(
        self,
        request: QualifiedCapabilityExecutionDispatchRequest,
    ) -> QualifiedCapabilityExecutionDispatchResult:
        task = self._task_lookup.resolve_task(request.task_id)
        if task is None:
            return QualifiedCapabilityExecutionDispatchResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                execution_request_id=request.execution_request_id,
                reason_detail="governed_execution_task_unavailable",
            )
        if task.task_id != request.task_id:
            return QualifiedCapabilityExecutionDispatchResult(
                disposition=QualifiedCapabilityExecutionDispatchDisposition.FAILED,
                execution_request_id=request.execution_request_id,
                reason_detail="governed_execution_task_id_mismatch",
            )
        token = bind_governed_execution_task(task)
        try:
            return await self._inner_async.dispatch_async(request)
        finally:
            reset_governed_execution_task(token)


__all__ = [
    "ActiveTaskRegistryGovernedExecutionTaskLookup",
    "GovernedExecutionTaskLookupPort",
    "GovernedTaskScopedQualifiedCapabilityExecutionDispatchService",
]
