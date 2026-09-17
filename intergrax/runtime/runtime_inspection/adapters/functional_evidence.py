# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Functional evidence persistence adapter — evidence references only."""

from __future__ import annotations

from intergrax.contracts.functional_evidence.persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidenceQueryRequest,
)
from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionEvidenceReference,
    RuntimeInspectionEvidenceSection,
)
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionEvidenceReadPort,
    RuntimeInspectionExecutionScope,
)


class FunctionalEvidenceInspectionAdapter(RuntimeInspectionEvidenceReadPort):
    def __init__(
        self,
        persistence: FunctionalEvidencePersistence,
        *,
        source_id: str = "functional_evidence",
        page_size: int = 100,
    ) -> None:
        self._persistence = persistence
        self._source_id = source_id
        self._page_size = page_size

    @property
    def source_id(self) -> str:
        return self._source_id

    def read_evidence_references(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionEvidenceSection:
        request = FunctionalEvidenceQueryRequest(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            attempt_id=scope.attempt_id,
            page_size=self._page_size,
        )
        page = self._persistence.query_evidence(request)
        references = tuple(
            RuntimeInspectionEvidenceReference(
                evidence_id=item.evidence_id,
                kind=item.kind.value,
                source_id=self._source_id,
            )
            for item in page.items
        )
        completeness = (
            RuntimeInspectionCompleteness.COMPLETE
            if references and page.next_cursor is None
            else RuntimeInspectionCompleteness.PARTIAL
        )
        if not references:
            completeness = RuntimeInspectionCompleteness.UNAVAILABLE
        return RuntimeInspectionEvidenceSection(
            references=references,
            completeness=completeness,
            source_id=self._source_id,
        )


__all__ = ["FunctionalEvidenceInspectionAdapter"]
