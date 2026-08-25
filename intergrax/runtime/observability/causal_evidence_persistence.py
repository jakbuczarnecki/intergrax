# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-owned persistence contract for canonical causal evidence (DIAG-1P1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence


class CausalEvidencePersistenceConflictError(Exception):
    """Raised when append encounters an existing evidence_id with different content."""


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

    @abstractmethod
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

    def close(self) -> None:
        """Release backend resources (no-op for most stores)."""
