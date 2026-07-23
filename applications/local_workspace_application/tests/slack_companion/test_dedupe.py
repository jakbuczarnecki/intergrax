# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

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


class _MissSyncDocumentStore(InMemoryDocumentStore):
    """Force two concurrent miss-gets to observe the unlocked claim race."""

    def __init__(self, parties: int = 2) -> None:
        super().__init__()
        self._barrier = threading.Barrier(parties)

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        result = super().get(partition_key, row_key)
        if result is None:
            self._barrier.wait(timeout=2.0)
        return result


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


def test_status_update_requires_matching_claim_token() -> None:
    repo = SlackEventDedupeRepository(InMemoryDocumentStore())
    claim = repo.claim(team_id="T1", event_id="EvToken")
    assert claim is not None
    repo.mark_completed(
        dedupe_key=claim.dedupe_key,
        claim_token="wrong-token",
        ask_run_id="should-not-stick",
    )
    # Wrong token left the record in processing; duplicate claim still blocked.
    assert repo.claim(team_id="T1", event_id="EvToken") is None
    repo.mark_completed(
        dedupe_key=claim.dedupe_key,
        claim_token=claim.claim_token,
        ask_run_id="ask-ok",
    )
    assert repo.claim(team_id="T1", event_id="EvToken") is None


def test_unlocked_claim_race_can_admit_two_owners() -> None:
    """Documents why get→put without a process lock is not exclusive."""
    store = _MissSyncDocumentStore(parties=2)
    repo_a = SlackEventDedupeRepository(store)
    repo_b = SlackEventDedupeRepository(store)
    results: list[SlackDedupeRecord | None] = []
    gate = threading.Barrier(2)

    def worker(repo: SlackEventDedupeRepository) -> None:
        gate.wait(timeout=2.0)
        # Bypass the process lock to reproduce the historical race.
        results.append(repo._claim_locked(dedupe_key="T1:EvConcurrent"))

    threads = [
        threading.Thread(target=worker, args=(repo_a,)),
        threading.Thread(target=worker, args=(repo_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    successes = [item for item in results if item is not None]
    assert len(results) == 2
    assert len(successes) == 2


def test_concurrent_claims_exactly_one_owner() -> None:
    store = InMemoryDocumentStore()
    repo_a = SlackEventDedupeRepository(store)
    repo_b = SlackEventDedupeRepository(store)
    results: list[SlackDedupeRecord | None] = []
    gate = threading.Barrier(2)

    def worker(repo: SlackEventDedupeRepository) -> None:
        gate.wait(timeout=2.0)
        results.append(repo.claim(team_id="T1", event_id="EvConcurrent"))

    threads = [
        threading.Thread(target=worker, args=(repo_a,)),
        threading.Thread(target=worker, args=(repo_b,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    successes = [item for item in results if item is not None]
    assert len(results) == 2
    assert len(successes) == 1
    assert results.count(None) == 1
