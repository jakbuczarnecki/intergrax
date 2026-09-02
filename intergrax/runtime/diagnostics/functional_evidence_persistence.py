# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform-owned persistence contract for functional/AI pipeline evidence (DIAG-FUNCTIONAL-2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PlatformFunctionalEvidence,
)


def functional_evidence_query_order_key(
    evidence: PlatformFunctionalEvidence,
) -> tuple[datetime, str]:
    """Canonical deterministic ordering for functional evidence queries."""
    return (evidence.provenance.recorded_at, str(evidence.evidence_id))


class FunctionalEvidencePersistenceError(Exception):
    """Base error for functional evidence persistence failures."""


class FunctionalEvidencePersistenceConflictError(FunctionalEvidencePersistenceError):
    """Raised when append encounters an existing evidence_id with different content."""


class FunctionalEvidencePersistenceIntegrityError(FunctionalEvidencePersistenceError):
    """Raised when stored evidence is outside the requested query scope."""


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceQueryRequest:
    """Bounded, tenant-scoped functional evidence query."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None = None
    kind: PipelineEvidenceKind | None = None
    page_size: int = 100
    cursor: str | None = None


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceQueryPage:
    """One bounded page of functional evidence facts."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    items: tuple[PlatformFunctionalEvidence, ...]
    next_cursor: str | None


class FunctionalEvidencePersistence(ABC):
    """
    Append-only store for ``PlatformFunctionalEvidence``.

    Implementations live behind this contract. Core diagnostics does not import
    vendor-specific storage backends.
    """

    @abstractmethod
    def append(self, evidence: PlatformFunctionalEvidence) -> PlatformFunctionalEvidence:
        """
        Persist a single functional evidence record.

        Idempotent on ``evidence_id``: duplicate append returns the original record.
        """

    @abstractmethod
    def query_evidence(self, request: FunctionalEvidenceQueryRequest) -> FunctionalEvidenceQueryPage:
        """
        Return one bounded page of evidence for an execution scope.

        Results are tenant-scoped and ordered by ``functional_evidence_query_order_key``.
        Pagination uses authenticated keyset cursors bound to the full query scope
        (tenant, task, run, optional attempt filter, optional kind filter).
        One traversal is a monotonic ordered scan; late evidence inserted before
        an already-consumed cursor may require a subsequent reconstruction cycle.
        """


__all__ = [
    "FunctionalEvidencePersistence",
    "FunctionalEvidencePersistenceConflictError",
    "FunctionalEvidencePersistenceError",
    "FunctionalEvidencePersistenceIntegrityError",
    "FunctionalEvidenceQueryPage",
    "FunctionalEvidenceQueryRequest",
    "functional_evidence_query_order_key",
]
