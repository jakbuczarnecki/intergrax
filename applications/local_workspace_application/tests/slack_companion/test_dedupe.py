# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from local_workspace_application.slack_companion.dedupe_repository import (
    DEDUPE_TTL_SECONDS,
    SlackEventDedupeRepository,
    build_slack_dedupe_key,
)
from local_workspace_application.slack_companion.models import (
    SlackDedupeRecord,
    SlackDedupeStatus,
)

pytestmark = pytest.mark.unit


def test_first_claim_succeeds() -> None:
    repo = SlackEventDedupeRepository(InMemoryDocumentStore())
    claim = repo.claim(team_id="T1", event_id="Ev1")
    assert claim is not None
    assert claim.dedupe_key == "T1:Ev1"
    assert claim.status is SlackDedupeStatus.PROCESSING


def test_duplicate_same_team_event_rejected() -> None:
    repo = SlackEventDedupeRepository(InMemoryDocumentStore())
    assert repo.claim(team_id="T1", event_id="Ev1") is not None
    assert repo.claim(team_id="T1", event_id="Ev1") is None


def test_same_event_id_different_team_separate_identity() -> None:
    repo = SlackEventDedupeRepository(InMemoryDocumentStore())
    assert repo.claim(team_id="T1", event_id="Ev1") is not None
    assert repo.claim(team_id="T2", event_id="Ev1") is not None


def test_expired_record_may_be_reclaimed() -> None:
    store = InMemoryDocumentStore()
    repo = SlackEventDedupeRepository(store)
    key = build_slack_dedupe_key(team_id="T1", event_id="Ev_old")
    past = datetime.now(timezone.utc) - timedelta(days=8)
    expired = SlackDedupeRecord(
        dedupe_key=key,
        status=SlackDedupeStatus.COMPLETED,
        claim_token="old",
        first_seen_at=past,
        updated_at=past,
        expires_at=past + timedelta(seconds=1),
    )
    store.put(
        DocumentRecord(
            partition_key="lkw.slack_companion:dedupe",
            row_key=key,
            data=expired.model_dump(mode="json"),
            ttl_seconds=DEDUPE_TTL_SECONDS,
        )
    )
    claim = repo.claim(team_id="T1", event_id="Ev_old")
    assert claim is not None
    assert claim.claim_token != "old"


def test_dedupe_key_format() -> None:
    assert build_slack_dedupe_key(team_id=" T1 ", event_id=" Ev9 ") == "T1:Ev9"
