# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Execution reconstruction adapter for runtime inspection federation."""

from __future__ import annotations

from intergrax.contracts.execution_reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructionReader,
)
from intergrax.contracts.execution_reconstruction_models import RuntimeHistoryCompleteness
from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
)
from intergrax.contracts.runtime_inspection.sections import RuntimeInspectionExecutionStateSection
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionExecutionFactsReader,
    RuntimeInspectionExecutionScope,
)


class ExecutionReconstructionInspectionAdapter(RuntimeInspectionExecutionFactsReader):
    def __init__(
        self,
        reconstructor: ExecutionReconstructionReader,
        *,
        source_id: str = "execution_reconstruction",
    ) -> None:
        self._reconstructor = reconstructor
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_execution_facts(
        self,
        scope: RuntimeInspectionExecutionScope,
    ):
        try:
            reconstruction = self._reconstructor.reconstruct_execution(
                scope.tenant_id,
                scope.task_id,
                scope.run_id,
            )
        except ExecutionReconstructionIntegrityError as exc:
            raise RuntimeInspectionError(
                RuntimeInspectionErrorCode.EXECUTION_FACTS_UNAVAILABLE,
                "execution reconstruction integrity failure",
                execution_id=scope.execution_id,
                source_id=self._source_id,
            ) from exc
        if reconstruction.tenant_id != scope.tenant_id:
            raise RuntimeInspectionError(
                RuntimeInspectionErrorCode.TENANT_BOUNDARY,
                "execution reconstruction tenant mismatch",
                execution_id=scope.execution_id,
                source_id=self._source_id,
            )
        return reconstruction


def execution_state_section(
    reconstruction,
    *,
    source_id: str,
) -> RuntimeInspectionExecutionStateSection:
    completeness = (
        RuntimeInspectionCompleteness.COMPLETE
        if reconstruction.runtime_history_completeness is RuntimeHistoryCompleteness.COMPLETE
        else RuntimeInspectionCompleteness.PARTIAL
    )
    if not reconstruction.has_runtime_events and not reconstruction.has_transport_evidence:
        completeness = RuntimeInspectionCompleteness.UNAVAILABLE
    return RuntimeInspectionExecutionStateSection(
        runtime_history_completeness=reconstruction.runtime_history_completeness.value,
        attempt_count=reconstruction.attempt_count,
        has_runtime_events=reconstruction.has_runtime_events,
        has_causal_evidence=reconstruction.has_transport_evidence,
        completeness=completeness,
        source_id=source_id,
    )


__all__ = [
    "ExecutionReconstructionInspectionAdapter",
    "execution_state_section",
]
