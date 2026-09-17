# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Artifact metadata read-model adapters for runtime inspection."""

from __future__ import annotations

from intergrax.contracts.execution_artifact_read import (
    ExecutionArtifactExecutionScope,
    ExecutionArtifactMetadataReadPort,
    ExecutionArtifactMetadataRecord,
)
from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
)
from intergrax.contracts.runtime_inspection.limits import (
    DEFAULT_RUNTIME_INSPECTION_ARTIFACT_LIMIT,
)
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionArtifactEntry,
    RuntimeInspectionArtifactSection,
)
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionArtifactReadPort,
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScope,
)
from intergrax.runtime.runtime_inspection.adapters.scope_integrity import validate_spine_event_scope
from intergrax.runtime.runtime_inspection.artifact_projection import (
    project_artifact_metadata_from_reconstruction,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def _to_artifact_scope(
    scope: RuntimeInspectionExecutionScope,
) -> ExecutionArtifactExecutionScope:
    return ExecutionArtifactExecutionScope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
        execution_id=scope.execution_id,
    )


def _validate_record_scope(
    scope: RuntimeInspectionExecutionScope,
    record: ExecutionArtifactMetadataRecord,
    *,
    source_id: str,
) -> None:
    if record.execution_id != scope.execution_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "artifact metadata execution_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )


class ReconstructionArtifactMetadataReader:
    """ExecutionArtifactMetadataReadPort backed by execution reconstruction facts."""

    def __init__(
        self,
        facts_reader: RuntimeInspectionExecutionFactsReader,
        *,
        source_id: str = "execution_artifact_reconstruction",
    ) -> None:
        self._facts_reader = facts_reader
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def list_artifact_metadata(
        self,
        scope: ExecutionArtifactExecutionScope,
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
                "artifact facts tenant mismatch",
                execution_id=scope.execution_id,
                source_id=self._source_id,
            )
        for positioned in reconstruction.positioned_events:
            validate_spine_event_scope(
                inspection_scope,
                positioned.event,
                source_id=self._source_id,
            )
        return project_artifact_metadata_from_reconstruction(reconstruction, limit=limit)


class ArtifactMetadataInspectionAdapter(RuntimeInspectionArtifactReadPort):
    def __init__(
        self,
        artifact_reader: ExecutionArtifactMetadataReadPort,
        *,
        artifact_limit: int = DEFAULT_RUNTIME_INSPECTION_ARTIFACT_LIMIT,
    ) -> None:
        self._artifact_reader = artifact_reader
        self._artifact_limit = artifact_limit

    @property
    def source_id(self) -> str:
        return self._artifact_reader.source_id

    def read_artifacts(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionArtifactSection:
        result = self._artifact_reader.list_artifact_metadata(
            _to_artifact_scope(scope),
            limit=self._artifact_limit,
        )
        artifacts: list[RuntimeInspectionArtifactEntry] = []
        for record in result.records:
            _validate_record_scope(scope, record, source_id=self.source_id)
            artifacts.append(
                RuntimeInspectionArtifactEntry(
                    artifact_ref=record.artifact_ref,
                    artifact_type=record.artifact_type,
                    lifecycle_status=record.lifecycle_status,
                    content_classification=record.content_classification,
                    execution_id=record.execution_id,
                    attempt_id=record.attempt_id,
                    sequence_key=record.sequence_key,
                    evidence_refs=record.evidence_refs,
                    safe_summary=sanitize_inspection_text(record.safe_summary),
                ),
            )
        completeness = (
            RuntimeInspectionCompleteness.COMPLETE
            if artifacts or not result.is_truncated
            else RuntimeInspectionCompleteness.PARTIAL
        )
        return RuntimeInspectionArtifactSection(
            artifacts=tuple(artifacts),
            is_truncated=result.is_truncated,
            completeness=completeness,
            source_id=self.source_id,
            source_available=True,
        )


__all__ = [
    "ArtifactMetadataInspectionAdapter",
    "ReconstructionArtifactMetadataReader",
]
