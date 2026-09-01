# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic bounded functional evidence reconstruction semantics (DIAG-FUNCTIONAL-1-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
)
from intergrax.contracts.functional_evidence_bounds import (
    MAX_SUPPORTING_EVIDENCE_REFS,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidenceQueryRequest,
)

_MONOTONIC_TRAVERSAL_LIMITATION = (
    "Reconstruction uses monotonic keyset traversal; evidence recorded before an "
    "already-consumed cursor during an in-flight scan may require a subsequent "
    "reconstruction cycle."
)


class FunctionalEvidenceReconstructionIntegrityError(Exception):
    """Raised when reconstruction input violates scope contracts."""


class FunctionalEvidenceCompletenessStatus(StrEnum):
    """Evidence completeness for a reconstruction scope — not diagnostic certainty."""

    NOT_EVALUATED = "not_evaluated"
    COMPLETE_FOR_REQUIREMENTS = "complete_for_requirements"
    INCOMPLETE_FOR_REQUIREMENTS = "incomplete_for_requirements"


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceCompleteness:
    """Per-kind evidence presence for one execution scope."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    present_kinds: frozenset[PipelineEvidenceKind]
    missing_kinds: frozenset[PipelineEvidenceKind]
    counts_by_kind: dict[PipelineEvidenceKind, int]

    @property
    def is_complete_for(self) -> frozenset[PipelineEvidenceKind]:
        return self.present_kinds

    @property
    def has_missing_evidence(self) -> bool:
        return bool(self.missing_kinds)


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceSummary:
    """Bounded aggregate over streamed functional evidence history."""

    total_evidence_count: int
    counts_by_kind: dict[PipelineEvidenceKind, int]
    first_recorded_at: datetime | None
    last_recorded_at: datetime | None


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceReconstruction:
    """
    Bounded derived projection over persisted functional evidence.

    NOT persisted and NOT a source of truth. Does not materialize full history.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    completeness_status: FunctionalEvidenceCompletenessStatus
    completeness: FunctionalEvidenceCompleteness
    evidence_summary: FunctionalEvidenceSummary
    supporting_evidence_refs: tuple[EventId, ...]
    limitations: tuple[str, ...]

    @property
    def has_missing_required_evidence(self) -> bool:
        return self.completeness_status is (
            FunctionalEvidenceCompletenessStatus.INCOMPLETE_FOR_REQUIREMENTS
        )


@dataclass(slots=True)
class _ReconstructionAccumulator:
    present_kinds: set[PipelineEvidenceKind]
    counts_by_kind: dict[PipelineEvidenceKind, int]
    supporting_refs: list[EventId]
    first_recorded_at: datetime | None
    last_recorded_at: datetime | None
    total_count: int

    @classmethod
    def empty(cls) -> _ReconstructionAccumulator:
        return cls(
            present_kinds=set(),
            counts_by_kind={},
            supporting_refs=[],
            first_recorded_at=None,
            last_recorded_at=None,
            total_count=0,
        )

    def observe(self, *, kind: PipelineEvidenceKind, recorded_at: datetime, evidence_id: EventId) -> None:
        self.total_count += 1
        self.present_kinds.add(kind)
        self.counts_by_kind[kind] = self.counts_by_kind.get(kind, 0) + 1
        if self.first_recorded_at is None or recorded_at < self.first_recorded_at:
            self.first_recorded_at = recorded_at
        if self.last_recorded_at is None or recorded_at > self.last_recorded_at:
            self.last_recorded_at = recorded_at
        if len(self.supporting_refs) < MAX_SUPPORTING_EVIDENCE_REFS:
            self.supporting_refs.append(evidence_id)


class FunctionalEvidenceReconstructor:
    """
    Deterministic bounded reconstruction from canonical functional evidence persistence.

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
        attempt_id: AttemptId | None = None,
        required_kinds: frozenset[PipelineEvidenceKind] = frozenset(),
        page_size: int = 1000,
    ) -> FunctionalEvidenceReconstruction:
        normalized_tenant = _require_tenant_id(tenant_id)
        accumulator = _ReconstructionAccumulator.empty()
        cursor: str | None = None
        while True:
            page = self._persistence.query_evidence(
                FunctionalEvidenceQueryRequest(
                    tenant_id=normalized_tenant,
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    page_size=page_size,
                    cursor=cursor,
                )
            )
            if page.tenant_id != normalized_tenant:
                raise FunctionalEvidenceReconstructionIntegrityError("tenant scope mismatch in page")
            if page.task_id != task_id or page.run_id != run_id:
                raise FunctionalEvidenceReconstructionIntegrityError("execution scope mismatch in page")
            for item in page.items:
                accumulator.observe(
                    kind=item.kind,
                    recorded_at=item.provenance.recorded_at,
                    evidence_id=item.evidence_id,
                )
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        present_kinds = frozenset(accumulator.present_kinds)
        missing_kinds = required_kinds - present_kinds
        if not required_kinds:
            completeness_status = FunctionalEvidenceCompletenessStatus.NOT_EVALUATED
        elif missing_kinds:
            completeness_status = FunctionalEvidenceCompletenessStatus.INCOMPLETE_FOR_REQUIREMENTS
        else:
            completeness_status = FunctionalEvidenceCompletenessStatus.COMPLETE_FOR_REQUIREMENTS

        return FunctionalEvidenceReconstruction(
            tenant_id=normalized_tenant,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            completeness_status=completeness_status,
            completeness=FunctionalEvidenceCompleteness(
                tenant_id=normalized_tenant,
                task_id=task_id,
                run_id=run_id,
                present_kinds=present_kinds,
                missing_kinds=missing_kinds,
                counts_by_kind=dict(accumulator.counts_by_kind),
            ),
            evidence_summary=FunctionalEvidenceSummary(
                total_evidence_count=accumulator.total_count,
                counts_by_kind=dict(accumulator.counts_by_kind),
                first_recorded_at=accumulator.first_recorded_at,
                last_recorded_at=accumulator.last_recorded_at,
            ),
            supporting_evidence_refs=tuple(accumulator.supporting_refs),
            limitations=(_MONOTONIC_TRAVERSAL_LIMITATION,),
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
    "FunctionalEvidenceCompletenessStatus",
    "FunctionalEvidenceReconstruction",
    "FunctionalEvidenceReconstructionIntegrityError",
    "FunctionalEvidenceReconstructor",
    "FunctionalEvidenceSummary",
]
