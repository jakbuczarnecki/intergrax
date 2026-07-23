# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed product dedupe for Slack Ask events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from intergrax.integrations.contracts.document_store import DocumentRecord, DocumentStore
from local_workspace_application.slack_companion.models import (
    SlackDedupeRecord,
    SlackDedupeStatus,
)

DEDUPE_TTL_SECONDS = 7 * 24 * 60 * 60
_PARTITION = "lkw.slack_companion:dedupe"


def build_slack_dedupe_key(*, team_id: str, event_id: str) -> str:
    return f"{team_id.strip()}:{event_id.strip()}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SlackEventDedupeRepository:
    """LKW-owned dedupe repository over the shared DocumentStore contract."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    def claim(self, *, team_id: str, event_id: str) -> SlackDedupeRecord | None:
        """Claim ownership of a product event.

        Returns the claim record when this caller owns processing; ``None`` for
        an active duplicate. Uses get → put → re-get with ``claim_token`` so a
        concurrent overwrite cannot produce two Ask owners.
        """
        dedupe_key = build_slack_dedupe_key(team_id=team_id, event_id=event_id)
        if not team_id.strip() or not event_id.strip():
            return None

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
        verified = self._get(dedupe_key)
        if verified is None or verified.claim_token != claim_token:
            return None
        if verified.expires_at <= now:
            return None
        return verified

    def mark_completed(
        self,
        *,
        dedupe_key: str,
        claim_token: str,
        ask_run_id: str | None = None,
    ) -> None:
        self._update_status(
            dedupe_key=dedupe_key,
            claim_token=claim_token,
            status=SlackDedupeStatus.COMPLETED,
            ask_run_id=ask_run_id,
        )

    def mark_failed(self, *, dedupe_key: str, claim_token: str) -> None:
        self._update_status(
            dedupe_key=dedupe_key,
            claim_token=claim_token,
            status=SlackDedupeStatus.FAILED,
            ask_run_id=None,
        )

    def _update_status(
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
