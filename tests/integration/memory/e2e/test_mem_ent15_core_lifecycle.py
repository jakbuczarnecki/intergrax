# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15: golden lifecycle — remember, projection, recall, supersession, temporal."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
)
from intergrax.memory.contracts.memory_models import MemoryKind
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
)
from intergrax.memory.contracts.memory_recall import MemorySupersessionIntent
from intergrax.memory.memory_temporal import is_memory_entry_active
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
from tests.integration.memory.e2e.harness import (
    FixedUtcTimeProvider,
    build_in_memory_memory_harness,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]

_AS_OF = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_golden_remember_canonical_entity_projection_recall_and_observability() -> None:
    harness = build_in_memory_memory_harness()
    identity = harness.identity()
    scope = harness.user_scope(identity)

    remembered = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="likes deterministic e2e", kind=MemoryKind.USER_FACT),
    )
    assert remembered.entry_id

    profile = await harness.canonical_profile()
    assert len(profile.memory_entries) == 1
    entry = profile.memory_entries[0]
    assert entry.content == "likes deterministic e2e"
    assert entry.revision >= 1

    entity = harness.entity_record_for_entry(harness.user_id, remembered.entry_id or "")
    assert entity is not None
    assert entity.source_memory_id == remembered.entry_id
    assert entity.source_memory_revision == entry.revision

    recall = await harness.plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="deterministic", top_k=5),
    )
    assert any(item.entry_id == remembered.entry_id for item in recall.items)

    cp_success = [
        event
        for event in harness.observability.events
        if event.operation is MemoryDiagnosticOperation.REMEMBER
        and event.outcome is MemoryDiagnosticOutcome.SUCCESS
    ]
    assert cp_success
    assert cp_success[0].tenant_id == harness.tenant_id


@pytest.mark.asyncio
async def test_supersession_revision_propagates_to_entity_projection() -> None:
    harness = build_in_memory_memory_harness(tenant_id="tenant-ent15-super", user_id="user-super")
    identity = harness.identity()
    scope = harness.user_scope(identity)

    older = await harness.plane.remember(
        identity, scope, MemoryControlRememberRequest(content="prefers tea", title="drink")
    )
    newer = await harness.plane.remember(
        identity, scope, MemoryControlRememberRequest(content="prefers coffee", title="drink")
    )
    assert older.entry_id and newer.entry_id

    await harness.plane.apply_memory_supersession(
        identity,
        scope,
        MemorySupersessionIntent(
            superseded_memory_id=older.entry_id,
            superseding_memory_id=newer.entry_id,
            reason="e2e-supersede",
        ),
    )

    profile = await harness.canonical_profile()
    by_id = {item.entry_id: item for item in profile.memory_entries}
    assert by_id[older.entry_id].lineage.superseded_by_memory_id == newer.entry_id
    assert by_id[older.entry_id].revision >= 2

    entity_old = harness.entity_record_for_entry(harness.user_id, older.entry_id)
    entity_new = harness.entity_record_for_entry(harness.user_id, newer.entry_id)
    assert entity_old is not None and entity_new is not None
    assert entity_old.source_memory_revision == by_id[older.entry_id].revision
    assert entity_new.source_memory_revision == by_id[newer.entry_id].revision


@pytest.mark.asyncio
async def test_temporal_validity_semantics_on_canonical_entries() -> None:
    FixedUtcTimeProvider.set_frozen(_AS_OF)
    harness = build_in_memory_memory_harness(tenant_id="tenant-ent15-temporal", user_id="user-temporal")
    identity = harness.identity()
    scope = harness.user_scope(identity)

    future_entry = UserProfileMemoryEntry(
        content="not yet valid",
        kind=MemoryKind.USER_FACT,
        valid_from="2030-01-01T00:00:00+00:00",
    )
    expired_entry = UserProfileMemoryEntry(
        content="expired fact",
        kind=MemoryKind.USER_FACT,
        valid_from="2020-01-01T00:00:00+00:00",
        valid_until="2025-01-01T00:00:00+00:00",
    )
    active_entry = UserProfileMemoryEntry(
        content="active fact",
        kind=MemoryKind.USER_FACT,
        valid_from="2020-01-01T00:00:00+00:00",
        valid_until="2030-01-01T00:00:00+00:00",
    )

    await harness.plane.remember(identity, scope, MemoryControlRememberRequest(entry=future_entry))
    await harness.plane.remember(identity, scope, MemoryControlRememberRequest(entry=expired_entry))
    active = await harness.plane.remember(
        identity, scope, MemoryControlRememberRequest(entry=active_entry)
    )

    profile = await harness.canonical_profile()
    stored = {e.entry_id: e for e in profile.memory_entries}
    assert not is_memory_entry_active(stored[future_entry.entry_id], as_of=_AS_OF)
    assert not is_memory_entry_active(stored[expired_entry.entry_id], as_of=_AS_OF)
    assert is_memory_entry_active(stored[active_entry.entry_id], as_of=_AS_OF)

    recall = await harness.plane.recall(identity, scope, MemoryControlRecallRequest(top_k=10))
    recalled_ids = {item.entry_id for item in recall.items}
    assert active.entry_id in recalled_ids
    assert future_entry.entry_id not in recalled_ids
    assert expired_entry.entry_id not in recalled_ids
