# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15: durable SQLite composition restart (A → B → C)."""

from __future__ import annotations

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
)
from tests.integration.memory.e2e.harness import (
    SqliteMemoryCompositionSession,
    build_sqlite_memory_harness,
    build_sqlite_memory_harness_from_path,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
async def test_sqlite_composition_restart_recall_and_continue_mutation() -> None:
    session = SqliteMemoryCompositionSession.create()
    tenant = "tenant-ent15-sqlite"
    user = "user-sqlite"

    composition_a = build_sqlite_memory_harness(
        db_path=session.db_path, tenant_id=tenant, user_id=user
    )
    identity = composition_a.identity()
    scope = composition_a.user_scope(identity)
    await composition_a.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="durability-marker-v1"),
    )
    await composition_a.close_store()

    composition_b = build_sqlite_memory_harness_from_path(
        session.db_path, tenant_id=tenant, user_id=user
    )
    recall_b = await composition_b.plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="durability", top_k=5),
    )
    assert any(item.content == "durability-marker-v1" for item in recall_b.items)

    await composition_b.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="durability-marker-v2"),
    )
    await composition_b.close_store()

    composition_c = build_sqlite_memory_harness_from_path(
        session.db_path, tenant_id=tenant, user_id=user
    )
    recall_c = await composition_c.plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(top_k=10),
    )
    contents = {item.content for item in recall_c.items}
    assert "durability-marker-v1" in contents
    assert "durability-marker-v2" in contents
    await composition_c.close_store()
    session.cleanup()
