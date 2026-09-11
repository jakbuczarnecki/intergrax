# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory prediction history store (tests, local lab) — PREDICTIVE R2."""

from __future__ import annotations

from dataclasses import replace
from threading import Lock

from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
    validate_outcome_lifecycle_transition,
)
from intergrax.runtime.prediction.history.predictive_history_persistence import (
    PredictiveHistoryPersistence,
    PredictiveHistoryPersistenceConflictError,
)


class InMemoryPredictiveHistoryPersistence(PredictiveHistoryPersistence):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], PredictiveRiskHistoryRecord] = {}
        self._lock = Lock()

    def append(self, record: PredictiveRiskHistoryRecord) -> PredictiveRiskHistoryRecord:
        key = (record.tenant_id, record.risk_signal_id)
        with self._lock:
            if key in self._records:
                raise PredictiveHistoryPersistenceConflictError(
                    f"history record exists: {record.risk_signal_id}",
                )
            self._records[key] = record
            return record

    def get(
        self,
        *,
        tenant_id: str,
        risk_signal_id: str,
    ) -> PredictiveRiskHistoryRecord | None:
        with self._lock:
            return self._records.get((tenant_id, risk_signal_id))

    def update_outcome(
        self,
        *,
        tenant_id: str,
        risk_signal_id: str,
        outcome_status: PredictiveHistoryOutcomeStatus,
    ) -> PredictiveRiskHistoryRecord:
        key = (tenant_id, risk_signal_id)
        with self._lock:
            current = self._records.get(key)
            if current is None:
                raise KeyError(f"history record missing: {risk_signal_id}")
            validate_outcome_lifecycle_transition(current.outcome_status, outcome_status)
            updated = replace(current, outcome_status=outcome_status)
            self._records[key] = updated
            return updated

    def list_for_tenant(
        self,
        *,
        tenant_id: str,
        subject_identity: str | None = None,
        limit: int = 100,
    ) -> tuple[PredictiveRiskHistoryRecord, ...]:
        if type(limit) is not int or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive int")
        with self._lock:
            rows = [
                record
                for (record_tenant, _), record in self._records.items()
                if record_tenant == tenant_id
            ]
        if subject_identity is not None:
            rows = [r for r in rows if r.subject_identity == subject_identity]
        rows.sort(key=lambda r: r.generated_at, reverse=True)
        return tuple(rows[:limit])


__all__ = ["InMemoryPredictiveHistoryPersistence"]
