# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""ToolRuntime read-model adapters for runtime inspection."""

from __future__ import annotations

from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
)
from intergrax.contracts.runtime_inspection.limits import (
    DEFAULT_RUNTIME_INSPECTION_TOOL_INVOCATION_LIMIT,
)
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionToolInvocation,
    RuntimeInspectionToolSection,
)
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScope,
    RuntimeInspectionToolReadPort,
)
from intergrax.contracts.tool_runtime_read import (
    ToolRuntimeExecutionScope,
    ToolRuntimeInvocationReadPort,
    ToolRuntimeInvocationRecord,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text
from intergrax.runtime.runtime_inspection.tool_projection import (
    project_tool_invocations_from_reconstruction,
)


def _to_tool_scope(scope: RuntimeInspectionExecutionScope) -> ToolRuntimeExecutionScope:
    return ToolRuntimeExecutionScope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        execution_id=scope.execution_id,
    )


def _validate_record_scope(
    scope: RuntimeInspectionExecutionScope,
    record: ToolRuntimeInvocationRecord,
    *,
    source_id: str,
) -> None:
    if record.execution_id != scope.execution_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "tool invocation execution_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )


class ReconstructionToolRuntimeInvocationReader:
    """ToolRuntimeInvocationReadPort backed by execution reconstruction facts."""

    def __init__(
        self,
        facts_reader: RuntimeInspectionExecutionFactsReader,
        *,
        source_id: str = "tool_runtime_reconstruction",
    ) -> None:
        self._facts_reader = facts_reader
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def list_invocations(
        self,
        scope: ToolRuntimeExecutionScope,
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
                "tool facts tenant mismatch",
                execution_id=scope.execution_id,
                source_id=self._source_id,
            )
        return project_tool_invocations_from_reconstruction(reconstruction, limit=limit)


class ToolRuntimeInvocationInspectionAdapter(RuntimeInspectionToolReadPort):
    def __init__(
        self,
        tool_reader: ToolRuntimeInvocationReadPort,
        *,
        invocation_limit: int = DEFAULT_RUNTIME_INSPECTION_TOOL_INVOCATION_LIMIT,
    ) -> None:
        self._tool_reader = tool_reader
        self._invocation_limit = invocation_limit

    @property
    def source_id(self) -> str:
        return self._tool_reader.source_id

    def read_tool_invocations(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionToolSection:
        tool_scope = _to_tool_scope(scope)
        result = self._tool_reader.list_invocations(
            tool_scope,
            limit=self._invocation_limit,
        )
        invocations: list[RuntimeInspectionToolInvocation] = []
        for record in result.records:
            _validate_record_scope(scope, record, source_id=self.source_id)
            invocations.append(
                RuntimeInspectionToolInvocation(
                    invocation_id=record.invocation_id,
                    tool_id=record.tool_id,
                    execution_id=record.execution_id,
                    attempt_id=record.attempt_id,
                    sequence_key=record.sequence_key,
                    outcome=record.outcome,
                    status_label=record.status_label,
                    failure_classification=record.failure_classification,
                    args_digest_ref=(
                        sanitize_inspection_text(record.args_digest_ref)
                        if record.args_digest_ref
                        else None
                    ),
                    provider_correlation_ref=record.provider_correlation_ref,
                    governance_evidence_refs=record.governance_evidence_refs,
                    evidence_refs=record.evidence_refs,
                    safe_summary=sanitize_inspection_text(record.safe_summary),
                ),
            )
        completeness = (
            RuntimeInspectionCompleteness.COMPLETE
            if invocations or not result.is_truncated
            else RuntimeInspectionCompleteness.PARTIAL
        )
        return RuntimeInspectionToolSection(
            invocations=tuple(invocations),
            is_truncated=result.is_truncated,
            completeness=completeness,
            source_id=self.source_id,
            source_available=True,
        )


__all__ = [
    "ReconstructionToolRuntimeInvocationReader",
    "ToolRuntimeInvocationInspectionAdapter",
]
