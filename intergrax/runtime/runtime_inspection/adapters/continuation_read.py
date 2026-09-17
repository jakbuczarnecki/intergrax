# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Execution continuation read adapters for runtime inspection."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
)
from intergrax.contracts.execution_continuation_read import ExecutionContinuationSnapshotReadPort
from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
)
from intergrax.contracts.runtime_inspection.sections import RuntimeInspectionContinuationSection
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionContinuationReadPort,
    RuntimeInspectionExecutionScope,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


class ExecutionContinuationInspectionAdapter(RuntimeInspectionContinuationReadPort):
    def __init__(self, continuation_reader: ExecutionContinuationSnapshotReadPort) -> None:
        self._continuation_reader = continuation_reader

    @property
    def source_id(self) -> str:
        return self._continuation_reader.source_id

    def read_continuation_state(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionContinuationSection:
        if scope.attempt_id is None:
            return RuntimeInspectionContinuationSection(
                completeness=RuntimeInspectionCompleteness.PARTIAL,
                source_id=self.source_id,
                source_available=True,
                is_durable=self._continuation_reader.is_durable,
            )
        identity = ExecutionContinuationIdentity(
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            execution_id=scope.execution_id,
        )
        pending = self._continuation_reader.read_current_episode(identity)
        if pending is None:
            return RuntimeInspectionContinuationSection(
                completeness=RuntimeInspectionCompleteness.COMPLETE,
                source_id=self.source_id,
                source_available=True,
                is_durable=self._continuation_reader.is_durable,
            )
        if pending.identity.execution_id != scope.execution_id:
            raise RuntimeInspectionError(
                RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
                "continuation execution_id mismatch",
                execution_id=scope.execution_id,
                source_id=self.source_id,
            )
        waiting = (
            pending.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
        )
        resolution_status = None
        if pending.lifecycle_state in (
            ExecutionContinuationLifecycleState.REJECTED,
            ExecutionContinuationLifecycleState.ESCALATED,
            ExecutionContinuationLifecycleState.CANCELLED,
        ):
            resolution_status = pending.lifecycle_state.value
        resume_status = None
        if pending.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED:
            resume_status = pending.lifecycle_state.value
        elif pending.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
            resume_status = "resume_authorized"
        pause_reason = None
        if pending.reason is not None:
            pause_reason = sanitize_inspection_text(pending.reason.value)
        approval_ref = pending.human_request_id
        return RuntimeInspectionContinuationSection(
            continuation_id=pending.continuation_id,
            lifecycle_state=pending.lifecycle_state,
            revision=pending.revision,
            pause_reason_classification=pause_reason,
            waiting_for_human=waiting,
            resolution_status=resolution_status,
            resume_status=resume_status,
            is_durable=self._continuation_reader.is_durable,
            approval_correlation_ref=approval_ref,
            completeness=RuntimeInspectionCompleteness.COMPLETE,
            source_id=self.source_id,
            source_available=True,
        )


__all__ = ["ExecutionContinuationInspectionAdapter"]
