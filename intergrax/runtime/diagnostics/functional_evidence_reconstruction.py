# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic functional evidence reconstruction semantics (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticCertainty
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidenceQueryRequest,
)


class FunctionalEvidenceReconstructionIntegrityError(Exception):
    """Raised when reconstruction input violates scope contracts."""


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceCompleteness:
    """Per-kind evidence presence for one execution scope."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    present_kinds: frozenset[PipelineEvidenceKind]
    missing_kinds: frozenset[PipelineEvidenceKind]

    @property
    def is_complete_for(self) -> frozenset[PipelineEvidenceKind]:
        return self.present_kinds

    @property
    def has_missing_evidence(self) -> bool:
        return bool(self.missing_kinds)


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceReconstruction:
    """
    Derived read model over persisted functional evidence.

    NOT persisted and NOT a source of truth.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    evidence: tuple[PlatformFunctionalEvidence, ...]
    completeness: FunctionalEvidenceCompleteness
    certainty: DiagnosticCertainty

    @property
    def has_insufficient_evidence(self) -> bool:
        return self.certainty is DiagnosticCertainty.INSUFFICIENT_EVIDENCE


class FunctionalEvidenceReconstructor:
    """
    Deterministic reconstruction from canonical functional evidence persistence.

    Does not infer missing stages, mutate execution state, or use LLM analysis.
    """

    def __init__(self, persistence: FunctionalEvidencePersistence) -> None:
        self._persistence = persistence

    def reconstruct(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        required_kinds: frozenset[PipelineEvidenceKind] = frozenset(),
        page_size: int = 1000,
    ) -> FunctionalEvidenceReconstruction:
        normalized_tenant = _require_tenant_id(tenant_id)
        collected: list[PlatformFunctionalEvidence] = []
        cursor: str | None = None
        while True:
            page = self._persistence.query_evidence(
                FunctionalEvidenceQueryRequest(
                    tenant_id=normalized_tenant,
                    task_id=task_id,
                    run_id=run_id,
                    page_size=page_size,
                    cursor=cursor,
                )
            )
            if page.tenant_id != normalized_tenant:
                raise FunctionalEvidenceReconstructionIntegrityError("tenant scope mismatch in page")
            if page.task_id != task_id or page.run_id != run_id:
                raise FunctionalEvidenceReconstructionIntegrityError("execution scope mismatch in page")
            collected.extend(page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        present_kinds = frozenset(item.kind for item in collected)
        missing_kinds = required_kinds - present_kinds
        certainty = (
            DiagnosticCertainty.INSUFFICIENT_EVIDENCE
            if missing_kinds
            else DiagnosticCertainty.PROVEN
        )
        return FunctionalEvidenceReconstruction(
            tenant_id=normalized_tenant,
            task_id=task_id,
            run_id=run_id,
            evidence=tuple(collected),
            completeness=FunctionalEvidenceCompleteness(
                tenant_id=normalized_tenant,
                task_id=task_id,
                run_id=run_id,
                present_kinds=present_kinds,
                missing_kinds=missing_kinds,
            ),
            certainty=certainty,
        )


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise FunctionalEvidenceReconstructionIntegrityError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise FunctionalEvidenceReconstructionIntegrityError("tenant_id must be non-empty")
    if tenant_id != normalized:
        raise FunctionalEvidenceReconstructionIntegrityError(
            "tenant_id must not contain leading or trailing whitespace",
        )
    return normalized


__all__ = [
    "FunctionalEvidenceCompleteness",
    "FunctionalEvidenceReconstruction",
    "FunctionalEvidenceReconstructionIntegrityError",
    "FunctionalEvidenceReconstructor",
]
