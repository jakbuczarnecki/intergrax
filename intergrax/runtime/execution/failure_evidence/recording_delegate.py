# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Delegate wrapper that records canonical EXECUTION_FAILED before propagating."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.contracts.execution_failure_evidence import (
    ExecutionFailureEvidenceRequest,
    ExecutionFailureKind,
)
from intergrax.contracts.execution_identity import require_active_execution_id
from intergrax.runtime.execution.boundary import ExecutionDelegate
from intergrax.runtime.execution.failure_evidence.active_context import (
    validate_active_execution_evidence_context,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")

_DELEGATE_FAILURE_SAFE_SUMMARY = "Execution delegate failed"


class ExecutionFailureRecordingDelegate(Generic[RequestT, ResultT]):
    """Records durable execution failure evidence for delegate exceptions."""

    __slots__ = ("_delegate",)

    def __init__(self, delegate: ExecutionDelegate[RequestT, ResultT]) -> None:
        self._delegate = delegate

    async def execute(self, request: RequestT) -> ResultT:
        try:
            return await self._delegate.execute(request)
        except Exception:
            self._record_delegate_failure()
            raise

    def _record_delegate_failure(self) -> None:
        from intergrax.runtime.execution.failure_evidence.active_context import (
            peek_active_execution_evidence_context,
        )

        if peek_active_execution_evidence_context() is None:
            return
        context = validate_active_execution_evidence_context()
        execution_id = require_active_execution_id()
        context.recorder.record_failure(
            ExecutionFailureEvidenceRequest(
                tenant_id=context.tenant_id,
                task_id=context.task_id,
                run_id=context.run_id,
                attempt_id=context.attempt_id,
                execution_id=execution_id,
                failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
                safe_summary=_DELEGATE_FAILURE_SAFE_SUMMARY,
                failure_code=None,
            ),
        )


def wrap_execution_delegate_for_failure_evidence(
    delegate: ExecutionDelegate[RequestT, ResultT],
) -> ExecutionDelegate[RequestT, ResultT]:
    return ExecutionFailureRecordingDelegate(delegate)


__all__ = [
    "ExecutionFailureRecordingDelegate",
    "wrap_execution_delegate_for_failure_evidence",
]
