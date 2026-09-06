# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-owned persistence contract for canonical causal evidence (DIAG-1P1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence

CAUSAL_EVIDENCE_QUERY_MAX_LIMIT = 1000


def validate_causal_evidence_query_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("causal_evidence_query_limit_invalid")
    if limit < 1 or limit > CAUSAL_EVIDENCE_QUERY_MAX_LIMIT:
        raise ValueError("causal_evidence_query_limit_invalid")
    return limit


def causal_evidence_query_order_key(
    evidence: PlatformCausalEvidence,
) -> tuple[datetime, str]:
    """Canonical deterministic ordering for list_for_execution / list_for_transport_task."""
    return (evidence.recorded_at, str(evidence.evidence_id))


@dataclass(frozen=True, slots=True)
class CausalEvidencePage:
    items: tuple[PlatformCausalEvidence, ...]
    next_cursor: str | None


class CausalEvidencePersistenceConflictError(Exception):
    """Raised when append encounters an existing evidence_id with different content."""


class CausalEvidencePersistenceIntegrityError(Exception):
    """Raised when indexed storage is inconsistent with the canonical causal record."""


def _consume_all_causal_evidence_pages(
    page_fn: Callable[..., CausalEvidencePage],
    *,
    tenant_id: str,
    limit: int = CAUSAL_EVIDENCE_QUERY_MAX_LIMIT,
    **scope_kwargs: str | TaskId | RunId,
) -> tuple[PlatformCausalEvidence, ...]:
    collected: list[PlatformCausalEvidence] = []
    cursor: str | None = None
    while True:
        page = page_fn(
            tenant_id=tenant_id,
            limit=limit,
            cursor=cursor,
            **scope_kwargs,
        )
        collected.extend(page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    return tuple(collected)


class CausalEvidencePersistence(ABC):
    """
    Append-only store for ``PlatformCausalEvidence``.

    Implementations (SQLite, Cassandra, …) live behind this contract.
    Producers and readers depend on the interface, not a specific backend.
    """

    @abstractmethod
    def append(self, evidence: PlatformCausalEvidence) -> PlatformCausalEvidence:
        """
        Persist a single causal evidence record.

        Idempotent on ``evidence_id``: duplicate append returns the original record.
        """

    @abstractmethod
    def page_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        limit: int,
        cursor: str | None = None,
    ) -> CausalEvidencePage:
        """Return one bounded page of execution-scoped causal evidence in canonical order."""

    @abstractmethod
    def page_for_transport_task(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> CausalEvidencePage:
        """Return one bounded page of transport-scoped causal evidence in canonical order."""

    def list_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> tuple[PlatformCausalEvidence, ...]:
        """
        Return causal evidence linking to the given runtime execution.

        Results are scoped by tenant. Missing data yields an empty tuple.
        """
        validated_task_id = validate_task_id(task_id)
        validated_run_id = validate_run_id(run_id)
        return _consume_all_causal_evidence_pages(
            self.page_for_execution,
            tenant_id=tenant_id,
            task_id=validated_task_id,
            run_id=validated_run_id,
        )

    def list_for_transport_task(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
    ) -> tuple[PlatformCausalEvidence, ...]:
        """
        Return causal evidence originating from the given transport task.

        ``transport_task_id`` is opaque provider transport identity — not runtime
        ``TaskId``. Results are scoped by tenant. Missing data yields an empty tuple.
        """
        return _consume_all_causal_evidence_pages(
            self.page_for_transport_task,
            tenant_id=tenant_id,
            provider=provider,
            transport_task_id=transport_task_id,
        )

    def close(self) -> None:
        """Release backend resources (no-op for most stores)."""
