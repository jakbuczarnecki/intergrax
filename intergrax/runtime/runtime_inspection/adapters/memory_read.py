# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Memory read-model adapters for runtime inspection."""

from __future__ import annotations

from intergrax.contracts.memory_runtime_read import (
    MemoryRuntimeExecutionScope,
    MemoryRuntimeOperationReadPort,
    MemoryRuntimeOperationRecord,
)
from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
)
from intergrax.contracts.runtime_inspection.limits import (
    DEFAULT_RUNTIME_INSPECTION_MEMORY_OPERATION_LIMIT,
)
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionMemoryOperation,
    RuntimeInspectionMemorySection,
)
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScope,
    RuntimeInspectionMemoryReadPort,
)
from intergrax.runtime.runtime_inspection.adapters.scope_integrity import validate_spine_event_scope
from intergrax.runtime.runtime_inspection.memory_projection import (
    project_memory_operations_from_reconstruction,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def _to_memory_scope(scope: RuntimeInspectionExecutionScope) -> MemoryRuntimeExecutionScope:
    return MemoryRuntimeExecutionScope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        execution_id=scope.execution_id,
    )


def _validate_record_scope(
    scope: RuntimeInspectionExecutionScope,
    record: MemoryRuntimeOperationRecord,
    *,
    source_id: str,
) -> None:
    if record.execution_id != scope.execution_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "memory operation execution_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )


class ReconstructionMemoryOperationReader:
    """MemoryRuntimeOperationReadPort backed by execution reconstruction facts."""

    def __init__(
        self,
        facts_reader: RuntimeInspectionExecutionFactsReader,
        *,
        source_id: str = "memory_runtime_reconstruction",
    ) -> None:
        self._facts_reader = facts_reader
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def list_operations(
        self,
        scope: MemoryRuntimeExecutionScope,
        *,
        limit: int,
    ):
        inspection_scope = RuntimeInspectionExecutionScope(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            execution_id=scope.execution_id,
        )
        reconstruction = self._facts_reader.read_execution_facts(inspection_scope)
        if reconstruction.tenant_id != scope.tenant_id:
            raise RuntimeInspectionError(
                RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
                "memory facts tenant mismatch",
                execution_id=scope.execution_id,
                source_id=self._source_id,
            )
        for positioned in reconstruction.positioned_events:
            validate_spine_event_scope(
                inspection_scope,
                positioned.event,
                source_id=self._source_id,
            )
        return project_memory_operations_from_reconstruction(reconstruction, limit=limit)


class MemoryOperationInspectionAdapter(RuntimeInspectionMemoryReadPort):
    def __init__(
        self,
        memory_reader: MemoryRuntimeOperationReadPort,
        *,
        operation_limit: int = DEFAULT_RUNTIME_INSPECTION_MEMORY_OPERATION_LIMIT,
    ) -> None:
        self._memory_reader = memory_reader
        self._operation_limit = operation_limit

    @property
    def source_id(self) -> str:
        return self._memory_reader.source_id

    def read_memory_operations(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionMemorySection:
        result = self._memory_reader.list_operations(
            _to_memory_scope(scope),
            limit=self._operation_limit,
        )
        operations: list[RuntimeInspectionMemoryOperation] = []
        for record in result.records:
            _validate_record_scope(scope, record, source_id=self.source_id)
            operations.append(
                RuntimeInspectionMemoryOperation(
                    operation_ref=record.operation_ref,
                    memory_class=record.memory_class,
                    operation_class=record.operation_class,
                    operation_status=record.operation_status,
                    record_ref=record.record_ref,
                    source_category=record.source_category,
                    execution_id=record.execution_id,
                    attempt_id=record.attempt_id,
                    sequence_key=record.sequence_key,
                    evidence_refs=record.evidence_refs,
                    safe_summary=sanitize_inspection_text(record.safe_summary),
                ),
            )
        completeness = (
            RuntimeInspectionCompleteness.COMPLETE
            if operations or not result.is_truncated
            else RuntimeInspectionCompleteness.PARTIAL
        )
        return RuntimeInspectionMemorySection(
            operations=tuple(operations),
            is_truncated=result.is_truncated,
            completeness=completeness,
            source_id=self.source_id,
            source_available=True,
        )


__all__ = [
    "MemoryOperationInspectionAdapter",
    "ReconstructionMemoryOperationReader",
]
