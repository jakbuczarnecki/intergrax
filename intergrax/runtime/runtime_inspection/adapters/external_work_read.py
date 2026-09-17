# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""External work read-model adapters for runtime inspection."""

from __future__ import annotations

from intergrax.contracts.external_work_runtime_read import (
    ExternalWorkRuntimeExecutionScope,
    ExternalWorkRuntimeFactReadPort,
)
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
)
from intergrax.contracts.runtime_inspection.limits import (
    DEFAULT_RUNTIME_INSPECTION_EXTERNAL_WORK_LIMIT,
)
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionExternalWorkEntry,
    RuntimeInspectionExternalWorkSection,
)
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScope,
    RuntimeInspectionExternalWorkReadPort,
)
from intergrax.runtime.runtime_inspection.adapters.scope_integrity import (
    validate_domain_record_scope,
    validate_spine_event_scope,
)
from intergrax.runtime.runtime_inspection.adapters.truncation_completeness import (
    completeness_for_read_result,
)
from intergrax.runtime.runtime_inspection.external_work_projection import (
    project_external_work_from_reconstruction,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def _to_external_work_scope(
    scope: RuntimeInspectionExecutionScope,
) -> ExternalWorkRuntimeExecutionScope:
    return ExternalWorkRuntimeExecutionScope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        execution_id=scope.execution_id,
    )


class ReconstructionExternalWorkFactReader:
    """ExternalWorkRuntimeFactReadPort backed by execution reconstruction facts."""

    def __init__(
        self,
        facts_reader: RuntimeInspectionExecutionFactsReader,
        *,
        source_id: str = "external_work_runtime_reconstruction",
    ) -> None:
        self._facts_reader = facts_reader
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def list_work_facts(
        self,
        scope: ExternalWorkRuntimeExecutionScope,
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
                "external work facts tenant mismatch",
                execution_id=scope.execution_id,
                source_id=self._source_id,
            )
        for positioned in reconstruction.positioned_events:
            validate_spine_event_scope(
                inspection_scope,
                positioned.event,
                source_id=self._source_id,
            )
        try:
            return project_external_work_from_reconstruction(reconstruction, limit=limit)
        except ValueError as exc:
            raise RuntimeInspectionError(
                RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
                "external work spine payload integrity failure",
                execution_id=scope.execution_id,
                source_id=self._source_id,
            ) from exc


class ExternalWorkInspectionAdapter(RuntimeInspectionExternalWorkReadPort):
    def __init__(
        self,
        external_work_reader: ExternalWorkRuntimeFactReadPort,
        *,
        work_limit: int = DEFAULT_RUNTIME_INSPECTION_EXTERNAL_WORK_LIMIT,
    ) -> None:
        self._external_work_reader = external_work_reader
        self._work_limit = work_limit

    @property
    def source_id(self) -> str:
        return self._external_work_reader.source_id

    def read_external_work(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionExternalWorkSection:
        result = self._external_work_reader.list_work_facts(
            _to_external_work_scope(scope),
            limit=self._work_limit,
        )
        entries: list[RuntimeInspectionExternalWorkEntry] = []
        for record in result.records:
            validate_domain_record_scope(
                scope,
                record,
                source_id=self.source_id,
                record_label="external work fact",
            )
            entries.append(
                RuntimeInspectionExternalWorkEntry(
                    work_ref=record.work_ref,
                    work_class=record.work_class,
                    work_status=record.work_status,
                    provider_ref=record.provider_ref,
                    failure_classification=record.failure_classification,
                    retryable=record.retryable,
                    execution_id=record.execution_id,
                    attempt_id=record.attempt_id,
                    sequence_key=record.sequence_key,
                    evidence_refs=record.evidence_refs,
                    safe_summary=sanitize_inspection_text(record.safe_summary),
                ),
            )
        completeness = completeness_for_read_result(result.is_truncated)
        return RuntimeInspectionExternalWorkSection(
            work_entries=tuple(entries),
            is_truncated=result.is_truncated,
            completeness=completeness,
            source_id=self.source_id,
            source_available=True,
        )


__all__ = [
    "ExternalWorkInspectionAdapter",
    "ReconstructionExternalWorkFactReader",
]
