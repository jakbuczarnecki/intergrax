# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""DocumentStore-backed prediction history (PREDICTIVE R2)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
    validate_outcome_lifecycle_transition,
)
from intergrax.runtime.prediction.history.predictive_history_persistence import (
    PredictiveHistoryPersistence,
    PredictiveHistoryPersistenceConflictError,
)
from intergrax.runtime.prediction.history.predictive_history_record_codec import (
    decode_predictive_history_json,
    encode_predictive_history_json,
)

_PARTITION_PREFIX = "intergrax.predictive_history.v1"
_RECORD_PREFIX = "record:"
_SUBJECT_INDEX_PREFIX = "subject:"


def _partition_key(tenant_id: str) -> str:
    return f"{_PARTITION_PREFIX}:{tenant_id}"


def _record_row_key(risk_signal_id: str) -> str:
    return f"{_RECORD_PREFIX}{risk_signal_id}"


def _subject_index_row_key(subject_identity: str, risk_signal_id: str) -> str:
    return f"{_SUBJECT_INDEX_PREFIX}{subject_identity}:{risk_signal_id}"


class DocumentStorePredictiveHistoryPersistence(PredictiveHistoryPersistence):
    """Tier-0 DocumentStore adapter — prediction outcomes only."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError("predictive history persistence requires ConditionalDocumentStore")
        self._store = document_store

    def append(self, record: PredictiveRiskHistoryRecord) -> PredictiveRiskHistoryRecord:
        partition = _partition_key(record.tenant_id)
        row_key = _record_row_key(record.risk_signal_id)
        encoded = encode_predictive_history_json(record)
        canonical = DocumentRecord(
            partition_key=partition,
            row_key=row_key,
            data={"history_json": encoded},
        )
        if not self._store.put_if_absent(canonical):
            raise PredictiveHistoryPersistenceConflictError(
                f"history record exists: {record.risk_signal_id}",
            )
        index_row = DocumentRecord(
            partition_key=partition,
            row_key=_subject_index_row_key(record.subject_identity, record.risk_signal_id),
            data={"risk_signal_id": record.risk_signal_id},
        )
        self._store.put_if_absent(index_row)
        return record

    def get(
        self,
        *,
        tenant_id: str,
        risk_signal_id: str,
    ) -> PredictiveRiskHistoryRecord | None:
        record = self._store.get(
            _partition_key(tenant_id),
            _record_row_key(risk_signal_id),
        )
        if record is None:
            return None
        raw = record.data.get("history_json")
        if raw is None:
            return None
        return decode_predictive_history_json(str(raw))

    def update_outcome(
        self,
        *,
        tenant_id: str,
        risk_signal_id: str,
        outcome_status: PredictiveHistoryOutcomeStatus,
    ) -> PredictiveRiskHistoryRecord:
        current = self.get(tenant_id=tenant_id, risk_signal_id=risk_signal_id)
        if current is None:
            raise KeyError(f"history record missing: {risk_signal_id}")
        validate_outcome_lifecycle_transition(current.outcome_status, outcome_status)
        updated = replace(current, outcome_status=outcome_status)
        partition = _partition_key(tenant_id)
        row_key = _record_row_key(risk_signal_id)
        expected = self._store.get(partition, row_key)
        if expected is None:
            raise KeyError(f"history record missing: {risk_signal_id}")
        replacement = DocumentRecord(
            partition_key=partition,
            row_key=row_key,
            data={"history_json": encode_predictive_history_json(updated)},
        )
        if not self._store.replace_if_match(expected=expected, replacement=replacement):
            raise PredictiveHistoryPersistenceConflictError(
                f"concurrent update: {risk_signal_id}",
            )
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
        partition = _partition_key(tenant_id)
        prefix = (
            f"{_SUBJECT_INDEX_PREFIX}{subject_identity}:"
            if subject_identity is not None
            else _RECORD_PREFIX
        )
        page = self._store.query(
            partition_key=partition,
            row_key_prefix=prefix,
            limit=limit * 4,
        )
        signal_ids: list[str] = []
        for item in page.records:
            if subject_identity is not None:
                signal_ids.append(str(item.data["risk_signal_id"]))
            else:
                if item.row_key.startswith(_RECORD_PREFIX):
                    signal_ids.append(item.row_key.removeprefix(_RECORD_PREFIX))
        rows: list[PredictiveRiskHistoryRecord] = []
        for signal_id in signal_ids:
            loaded = self.get(tenant_id=tenant_id, risk_signal_id=signal_id)
            if loaded is not None:
                rows.append(loaded)
        rows.sort(key=lambda r: r.generated_at, reverse=True)
        return tuple(rows[:limit])


__all__ = ["DocumentStorePredictiveHistoryPersistence"]
