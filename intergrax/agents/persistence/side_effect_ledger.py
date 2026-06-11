# © Artur Czarnecki. All rights reserved.

"""In-run side-effect ledger with idempotency dedupe (ACP-PROD-2)."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from intergrax.contracts.side_effect import (
    SideEffectKind,
    SideEffectRecord,
    SideEffectStatus,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SideEffectLedger:
    """Tracks side effects for checkpoint resume and exactly-once dedupe."""

    def __init__(self, records: list[SideEffectRecord] | None = None) -> None:
        self._records: list[SideEffectRecord] = list(records or [])
        self._committed_keys: set[str] = {
            record.idempotency_key
            for record in self._records
            if record.status == SideEffectStatus.COMMITTED
        }

    def records(self) -> list[SideEffectRecord]:
        return list(self._records)

    def is_committed(self, idempotency_key: str) -> bool:
        return idempotency_key in self._committed_keys

    def register(
        self,
        *,
        idempotency_key: str,
        run_id: str,
        step_index: int,
        target: str,
        kind: SideEffectKind = SideEffectKind.TOOL,
        side_effect_id: str | None = None,
    ) -> SideEffectRecord:
        if self.is_committed(idempotency_key):
            for record in self._records:
                if record.idempotency_key == idempotency_key:
                    return record
        record = SideEffectRecord(
            side_effect_id=side_effect_id or f"sfx_{uuid4().hex}",
            idempotency_key=idempotency_key,
            run_id=run_id,
            step_index=step_index,
            kind=kind,
            target=target,
            status=SideEffectStatus.PENDING,
        )
        self._records.append(record)
        return record

    def commit(
        self,
        idempotency_key: str,
        *,
        external_ref: str | None = None,
        committed_externally: bool = False,
    ) -> SideEffectRecord | None:
        for index, record in enumerate(self._records):
            if record.idempotency_key != idempotency_key:
                continue
            updated = record.model_copy(
                update={
                    "status": SideEffectStatus.COMMITTED,
                    "committed_at": _utc_now(),
                    "external_ref": external_ref,
                    "committed_externally": committed_externally,
                }
            )
            self._records[index] = updated
            self._committed_keys.add(idempotency_key)
            return updated
        return None

    def should_skip_replay(self, idempotency_key: str) -> bool:
        return self.is_committed(idempotency_key)

    def committed_for_step(self, step_index: int) -> list[SideEffectRecord]:
        return [
            record
            for record in self._records
            if record.step_index == step_index and record.status == SideEffectStatus.COMMITTED
        ]

    def mark_failed(self, idempotency_key: str) -> SideEffectRecord | None:
        for index, record in enumerate(self._records):
            if record.idempotency_key != idempotency_key:
                continue
            updated = record.model_copy(update={"status": SideEffectStatus.FAILED})
            self._records[index] = updated
            self._committed_keys.discard(idempotency_key)
            return updated
        return None

    def mark_compensated(self, idempotency_key: str) -> SideEffectRecord | None:
        for index, record in enumerate(self._records):
            if record.idempotency_key != idempotency_key:
                continue
            updated = record.model_copy(update={"status": SideEffectStatus.COMPENSATED})
            self._records[index] = updated
            self._committed_keys.discard(idempotency_key)
            return updated
        return None
