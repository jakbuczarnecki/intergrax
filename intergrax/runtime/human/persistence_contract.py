# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 persistence contract for canonical human decision records (§42.38)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from intergrax.runtime.human.models import HumanDecisionRecord, HumanResponseVerdict


class HumanDecisionPersistence(ABC):
    """
    Persisted human decision store.

    Implementations (SQLite, Postgres, …) live behind this contract.
    Nexus, HITL tools, and debug surfaces depend on the interface, not a backend.
    """

    @abstractmethod
    def record(self, record: HumanDecisionRecord) -> HumanDecisionRecord:
        """Persist a human decision record."""

    @abstractmethod
    def list_for_task(self, task_id: str, tenant_id: str) -> list[HumanDecisionRecord]:
        """Return decisions for a task scoped by tenant (oldest first)."""

    @abstractmethod
    def list_escalations(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
    ) -> list[HumanDecisionRecord]:
        """Return escalation verdict decisions for a tenant (newest first)."""

    @abstractmethod
    def get_decision(self, decision_id: str, tenant_id: str) -> HumanDecisionRecord | None:
        ...

    @abstractmethod
    def summarize_queue(self, tenant_id: str) -> dict[str, int]:
        """Return verdict counts grouped by verdict value for a tenant."""

    def close(self) -> None:
        """Release backend resources (no-op for most stores)."""


class InMemoryHumanDecisionPersistence(HumanDecisionPersistence):
    """In-memory backend for tests and vendor-neutral dependency proofs."""

    def __init__(self) -> None:
        self._records: dict[str, HumanDecisionRecord] = {}
        self._task_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    def record(self, record: HumanDecisionRecord) -> HumanDecisionRecord:
        self._records[record.decision_id] = record
        key = (record.task_id, record.tenant_id)
        if record.decision_id not in self._task_index[key]:
            self._task_index[key].append(record.decision_id)
        return record

    def list_for_task(self, task_id: str, tenant_id: str) -> list[HumanDecisionRecord]:
        ids = self._task_index.get((task_id, tenant_id), [])
        records = [self._records[decision_id] for decision_id in ids if decision_id in self._records]
        return sorted(records, key=lambda item: item.created_at_utc)

    def list_escalations(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
    ) -> list[HumanDecisionRecord]:
        escalations = [
            record
            for record in self._records.values()
            if record.tenant_id == tenant_id and record.verdict is HumanResponseVerdict.ESCALATE
        ]
        escalations.sort(key=lambda item: item.created_at_utc, reverse=True)
        return escalations[:limit]

    def get_decision(self, decision_id: str, tenant_id: str) -> HumanDecisionRecord | None:
        record = self._records.get(decision_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record

    def summarize_queue(self, tenant_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            if record.tenant_id != tenant_id:
                continue
            verdict = record.verdict.value
            counts[verdict] = counts.get(verdict, 0) + 1
        return counts
