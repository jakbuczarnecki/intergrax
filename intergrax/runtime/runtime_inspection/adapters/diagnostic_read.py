# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""DiagnosticReadService adapter for runtime inspection federation."""

from __future__ import annotations

from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionDiagnosticFinding,
    RuntimeInspectionDiagnosticSection,
)
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionDiagnosticReadPort,
    RuntimeInspectionExecutionScope,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import DiagnosticReadIntegrityError
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


class DiagnosticReadServiceInspectionAdapter(RuntimeInspectionDiagnosticReadPort):
    def __init__(
        self,
        diagnostic_read_service: DiagnosticReadService,
        *,
        source_id: str = "diagnostic_read_service",
    ) -> None:
        self._service = diagnostic_read_service
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_diagnostics(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionDiagnosticSection:
        try:
            assessment = self._service.assess_execution_scope_for_inspection(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
            )
        except DiagnosticReadIntegrityError:
            return RuntimeInspectionDiagnosticSection(
                completeness=RuntimeInspectionCompleteness.UNAVAILABLE,
                source_id=self._source_id,
                limitations=("diagnostic_read_integrity",),
            )
        if assessment is None:
            return RuntimeInspectionDiagnosticSection(
                completeness=RuntimeInspectionCompleteness.UNAVAILABLE,
                source_id=self._source_id,
                limitations=("execution_evidence_unavailable",),
            )
        findings = tuple(
            RuntimeInspectionDiagnosticFinding(
                kind=finding.kind.value,
                certainty=finding.certainty.value,
                safe_summary=sanitize_inspection_text(finding.claim),
            )
            for finding in assessment.findings
        )
        limitations = tuple(
            sanitize_inspection_text(limitation.factual_message)
            for limitation in assessment.limitations
        )
        completeness = (
            RuntimeInspectionCompleteness.COMPLETE
            if findings or limitations
            else RuntimeInspectionCompleteness.PARTIAL
        )
        return RuntimeInspectionDiagnosticSection(
            findings=findings,
            limitations=limitations,
            completeness=completeness,
            source_id=self._source_id,
        )


__all__ = ["DiagnosticReadServiceInspectionAdapter"]
