# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed product dedupe for Slack Ask events."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from local_workspace_application.slack_companion.models import (
    SlackDedupeRecord,
    SlackDedupeStatus,
)

DEDUPE_TTL_SECONDS = 7 * 24 * 60 * 60
_PARTITION = "lkw.slack_companion:dedupe"

# Single-process MVP: shared by all repository instances in this process.
# Not a distributed lock — multi-process / HA deployments are out of scope.
_PROCESS_CLAIM_LOCK = threading.Lock()


def build_slack_dedupe_key(*, team_id: str, event_id: str) -> str:
    return f"{team_id.strip()}:{event_id.strip()}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SlackEventDedupeRepository:
    """LKW-owned dedupe repository over the shared DocumentStore contract.

    Claim exclusivity for the current frozen MVP model (single local process) is
    enforced by a process-wide lock around the full read → evaluate → write
    sequence. The shared ``DocumentStore`` contract has no create-if-absent /
    compare-and-set operation.
    """

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    def claim(self, *, team_id: str, event_id: str) -> SlackDedupeRecord | None:
        """Claim ownership of a product event.

        Returns the claim record when this caller owns processing; ``None`` for
        an active duplicate. Exactly one successful claim per active key is
        guaranteed within a single local process.
        """
        if not team_id.strip() or not event_id.strip():
            return None

        dedupe_key = build_slack_dedupe_key(team_id=team_id, event_id=event_id)
        with _PROCESS_CLAIM_LOCK:
            return self._claim_locked(dedupe_key=dedupe_key)

    def mark_completed(
        self,
        *,
        dedupe_key: str,
        claim_token: str,
        ask_run_id: str | None = None,
    ) -> None:
        with _PROCESS_CLAIM_LOCK:
            self._update_status_locked(
                dedupe_key=dedupe_key,
                claim_token=claim_token,
                status=SlackDedupeStatus.COMPLETED,
                ask_run_id=ask_run_id,
            )

    def mark_failed(self, *, dedupe_key: str, claim_token: str) -> None:
        with _PROCESS_CLAIM_LOCK:
            self._update_status_locked(
                dedupe_key=dedupe_key,
                claim_token=claim_token,
                status=SlackDedupeStatus.FAILED,
                ask_run_id=None,
            )

    def _claim_locked(self, *, dedupe_key: str) -> SlackDedupeRecord | None:
        existing = self._get(dedupe_key)
        now = _utcnow()
        if existing is not None and existing.expires_at > now:
            return None

        claim_token = uuid4().hex
        record = SlackDedupeRecord(
            dedupe_key=dedupe_key,
            status=SlackDedupeStatus.PROCESSING,
            claim_token=claim_token,
            first_seen_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=DEDUPE_TTL_SECONDS),
        )
        self._put(record)
        return record

    def _update_status_locked(
        self,
        *,
        dedupe_key: str,
        claim_token: str,
        status: SlackDedupeStatus,
        ask_run_id: str | None,
    ) -> None:
        current = self._get(dedupe_key)
        if current is None or current.claim_token != claim_token:
            return
        updated = current.model_copy(
            update={
                "status": status,
                "updated_at": _utcnow(),
                "ask_run_id": ask_run_id if ask_run_id is not None else current.ask_run_id,
            }
        )
        self._put(updated)

    def _get(self, dedupe_key: str) -> SlackDedupeRecord | None:
        record = self._store.get(_PARTITION, dedupe_key)
        if record is None:
            return None
        parsed = SlackDedupeRecord.model_validate(dict(record.data))
        if parsed.expires_at <= _utcnow():
            return None
        return parsed

    def _put(self, record: SlackDedupeRecord) -> None:
        self._store.put(
            DocumentRecord(
                partition_key=_PARTITION,
                row_key=record.dedupe_key,
                data=record.model_dump(mode="json"),
                ttl_seconds=DEDUPE_TTL_SECONDS,
            )
        )
