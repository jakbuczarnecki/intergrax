# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory causal evidence persistence (tests, conformance, local lab)."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import DefaultDict

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
    CausalEvidencePersistenceConflictError,
)


class InMemoryCausalEvidencePersistence(CausalEvidencePersistence):
    def __init__(self) -> None:
        self._accepted_by_evidence_id: dict[str, PlatformCausalEvidence] = {}
        self._insertion_order: list[str] = []
        self._by_execution: DefaultDict[tuple[str, str, str], list[str]] = defaultdict(list)
        self._by_transport: DefaultDict[tuple[str, str, str], list[str]] = defaultdict(list)
        self._lock = Lock()

    def append(self, evidence: PlatformCausalEvidence) -> PlatformCausalEvidence:
        with self._lock:
            existing = self._accepted_by_evidence_id.get(evidence.evidence_id)
            if existing is not None:
                if existing != evidence:
                    raise CausalEvidencePersistenceConflictError(
                        "conflicting causal evidence for evidence_id",
                    )
                return existing
            self._accepted_by_evidence_id[evidence.evidence_id] = evidence
            self._insertion_order.append(evidence.evidence_id)
            execution_key = (
                evidence.tenant_id,
                evidence.target.task_id,
                evidence.target.run_id,
            )
            transport_key = (
                evidence.tenant_id,
                evidence.source.provider,
                evidence.source.task_id,
            )
            self._by_execution[execution_key].append(evidence.evidence_id)
            self._by_transport[transport_key].append(evidence.evidence_id)
            return evidence

    def list_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> tuple[PlatformCausalEvidence, ...]:
        key = (tenant_id, task_id, run_id)
        return self._resolve_ids(self._by_execution.get(key, []))

    def list_for_transport_task(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
    ) -> tuple[PlatformCausalEvidence, ...]:
        key = (tenant_id, provider, transport_task_id)
        return self._resolve_ids(self._by_transport.get(key, []))

    def _resolve_ids(self, evidence_ids: list[str]) -> tuple[PlatformCausalEvidence, ...]:
        ordered = [
            self._accepted_by_evidence_id[evidence_id]
            for evidence_id in self._insertion_order
            if evidence_id in evidence_ids
        ]
        return tuple(ordered)
