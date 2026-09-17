# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Model read-model adapters for runtime inspection."""

from __future__ import annotations

from intergrax.contracts.model_runtime_read import (
    ModelRuntimeExecutionScope,
    ModelRuntimeInvocationReadPort,
)
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
)
from intergrax.contracts.runtime_inspection.limits import (
    DEFAULT_RUNTIME_INSPECTION_MODEL_INVOCATION_LIMIT,
)
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionModelInvocation,
    RuntimeInspectionModelSection,
)
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScope,
    RuntimeInspectionModelReadPort,
)
from intergrax.runtime.runtime_inspection.adapters.scope_integrity import (
    validate_domain_record_scope,
    validate_spine_event_scope,
)
from intergrax.runtime.runtime_inspection.adapters.truncation_completeness import (
    completeness_for_read_result,
)
from intergrax.runtime.runtime_inspection.model_projection import (
    project_model_invocations_from_reconstruction,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def _to_model_scope(scope: RuntimeInspectionExecutionScope) -> ModelRuntimeExecutionScope:
    return ModelRuntimeExecutionScope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        execution_id=scope.execution_id,
    )


class ReconstructionModelInvocationReader:
    """ModelRuntimeInvocationReadPort backed by execution reconstruction facts."""

    def __init__(
        self,
        facts_reader: RuntimeInspectionExecutionFactsReader,
        *,
        source_id: str = "model_runtime_reconstruction",
    ) -> None:
        self._facts_reader = facts_reader
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def list_invocations(
        self,
        scope: ModelRuntimeExecutionScope,
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
                "model facts tenant mismatch",
                execution_id=scope.execution_id,
                source_id=self._source_id,
            )
        for positioned in reconstruction.positioned_events:
            validate_spine_event_scope(
                inspection_scope,
                positioned.event,
                source_id=self._source_id,
            )
        return project_model_invocations_from_reconstruction(reconstruction, limit=limit)


class ModelInvocationInspectionAdapter(RuntimeInspectionModelReadPort):
    def __init__(
        self,
        model_reader: ModelRuntimeInvocationReadPort,
        *,
        invocation_limit: int = DEFAULT_RUNTIME_INSPECTION_MODEL_INVOCATION_LIMIT,
    ) -> None:
        self._model_reader = model_reader
        self._invocation_limit = invocation_limit

    @property
    def source_id(self) -> str:
        return self._model_reader.source_id

    def read_model_invocations(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionModelSection:
        result = self._model_reader.list_invocations(
            _to_model_scope(scope),
            limit=self._invocation_limit,
        )
        invocations: list[RuntimeInspectionModelInvocation] = []
        for record in result.records:
            validate_domain_record_scope(
                scope,
                record,
                source_id=self.source_id,
                record_label="model invocation",
            )
            invocations.append(
                RuntimeInspectionModelInvocation(
                    invocation_ref=record.invocation_ref,
                    model_ref=sanitize_inspection_text(record.model_ref),
                    capability_label=sanitize_inspection_text(record.capability_label),
                    invocation_status=record.invocation_status,
                    prompt_tokens=record.prompt_tokens,
                    completion_tokens=record.completion_tokens,
                    total_tokens=record.total_tokens,
                    finish_reason=record.finish_reason,
                    execution_id=record.execution_id,
                    attempt_id=record.attempt_id,
                    sequence_key=record.sequence_key,
                    evidence_refs=record.evidence_refs,
                    safe_summary=sanitize_inspection_text(record.safe_summary),
                ),
            )
        completeness = completeness_for_read_result(result.is_truncated)
        return RuntimeInspectionModelSection(
            invocations=tuple(invocations),
            is_truncated=result.is_truncated,
            completeness=completeness,
            source_id=self.source_id,
            source_available=True,
        )


__all__ = [
    "ModelInvocationInspectionAdapter",
    "ReconstructionModelInvocationReader",
]
