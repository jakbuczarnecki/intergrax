# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for InMemoryConversationalMemoryStore.

These tests define the behavioral contract for the in-memory persistence provider:
- loading returns a fully initialized ConversationalMemory aggregate,
- saving uses defensive copying (store state is not affected by later mutations),
- append_message updates both the aggregate and persisted storage,
- delete_session provides no-error semantics (idempotent),
- session listing reflects persisted state.

Why this matters:
The store is commonly used in local development, prototyping, and tests.
Regressions here can silently corrupt session history and break higher-level flows.
"""

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.memory.stores.in_memory_conversational_store import (
    InMemoryConversationalMemoryStore,
)


pytestmark = pytest.mark.unit
TENANT = "test-tenant"

@pytest.mark.asyncio
async def test_load_memory_unknown_session_returns_empty_memory() -> None:
    """
    Loading a missing session must return an empty ConversationalMemory instance
    with the requested session_id.
    """
    store = InMemoryConversationalMemoryStore()

    mem = await store.load_memory(tenant_id=TENANT, session_id= "missing-session")

    assert mem.session_id == "missing-session"
    assert mem.get_all() == []


@pytest.mark.asyncio
async def test_save_memory_persists_full_history_with_defensive_copying() -> None:
    """
    save_memory() must persist the entire aggregate state and use defensive copying.

    Contract:
    - After save_memory(), a subsequent load_memory() returns the saved messages.
    - Mutating the in-memory aggregate after saving must NOT affect persisted state.
    """
    store = InMemoryConversationalMemoryStore()

    mem = ConversationalMemory(session_id="s1")
    mem.add("user", "u1")
    mem.add("assistant", "a1")

    await store.save_memory(tenant_id=TENANT, memory=mem)

    # Mutate the aggregate AFTER saving.
    mem.add("user", "u2")

    # Persisted state must remain as it was at save time.
    reloaded = await store.load_memory(tenant_id=TENANT, session_id= "s1")
    assert [m.content for m in reloaded.get_all()] == ["u1", "a1"]


@pytest.mark.asyncio
async def test_append_message_updates_aggregate_and_persists_incrementally() -> None:
    """
    append_message() must:
    - apply aggregate mutation (including trimming policy),
    - and persist the newly appended message in store state.

    This is a critical invariant for incremental persistence backends.
    """
    store = InMemoryConversationalMemoryStore()

    mem = await store.load_memory(tenant_id=TENANT, session_id= "s1")
    assert mem.get_all() == []

    msg = ChatMessage(role="user", content="hello")
    await store.append_message(tenant_id=TENANT, memory=mem, message=msg)

    # Aggregate updated.
    assert [m.content for m in mem.get_all()] == ["hello"]

    # Persisted state updated.
    reloaded = await store.load_memory(tenant_id=TENANT, session_id= "s1")
    assert [m.content for m in reloaded.get_all()] == ["hello"]


@pytest.mark.asyncio
async def test_load_memory_respects_max_messages_via_aggregate_trimming() -> None:
    """
    load_memory(max_messages=N) should enforce the limit via the aggregate.

    The store loads persisted messages and extends the aggregate; trimming is
    the responsibility of ConversationalMemory.
    """
    store = InMemoryConversationalMemoryStore()

    mem = ConversationalMemory(session_id="s1")
    mem.add("user", "m1")
    mem.add("user", "m2")
    mem.add("user", "m3")
    mem.add("user", "m4")

    await store.save_memory(tenant_id=TENANT, memory=mem)

    limited = await store.load_memory(tenant_id=TENANT, session_id= "s1", max_messages=2)
    assert [m.content for m in limited.get_all()] == ["m3", "m4"]


@pytest.mark.asyncio
async def test_delete_session_is_idempotent_and_has_no_error_semantics() -> None:
    """
    delete_session() must be safe and idempotent.

    Contract:
    - Deleting an existing session removes its persisted data.
    - Deleting a missing session does not raise.
    """
    store = InMemoryConversationalMemoryStore()

    mem = ConversationalMemory(session_id="s1")
    mem.add("user", "x")
    await store.save_memory(tenant_id=TENANT, memory=mem)

    await store.delete_session(tenant_id=TENANT, session_id= "s1")
    reloaded = await store.load_memory(tenant_id=TENANT, session_id= "s1")
    assert reloaded.get_all() == []

    # Idempotent: deleting again must not raise.
    await store.delete_session(tenant_id=TENANT, session_id= "s1")
    reloaded2 = await store.load_memory(tenant_id=TENANT, session_id= "s1")
    assert reloaded2.get_all() == []


@pytest.mark.asyncio
async def test_list_sessions_reflects_persisted_state() -> None:
    """
    list_sessions() should reflect which session IDs currently have persisted data.
    """
    store = InMemoryConversationalMemoryStore()

    assert store.list_sessions(tenant_id=TENANT) == []

    mem1 = ConversationalMemory(session_id="s1")
    mem1.add("user", "a")
    await store.save_memory(tenant_id=TENANT, memory=mem1)

    mem2 = ConversationalMemory(session_id="s2")
    mem2.add("user", "b")
    await store.save_memory(tenant_id=TENANT, memory=mem2)

    sessions = set(store.list_sessions(tenant_id=TENANT))
    assert sessions == {"s1", "s2"}

    await store.delete_session(tenant_id=TENANT, session_id= "s1")
    sessions2 = set(store.list_sessions(tenant_id=TENANT))
    assert sessions2 == {"s2"}

@pytest.mark.asyncio
async def test_cross_tenant_isolation_for_same_session_id() -> None:
    """
    Same session_id used by different tenants must be fully isolated.
    """

    store = InMemoryConversationalMemoryStore()

    tenant_a = "tenant-a"
    tenant_b = "tenant-b"
    session_id = "shared-session"

    # --- Tenant A ---
    mem_a = ConversationalMemory(session_id=session_id)
    mem_a.add("user", "message-from-A")
    await store.save_memory(
        tenant_id=tenant_a,
        memory=mem_a,
    )

    # --- Tenant B ---
    mem_b = ConversationalMemory(session_id=session_id)
    mem_b.add("user", "message-from-B")
    await store.save_memory(
        tenant_id=tenant_b,
        memory=mem_b,
    )

    # --- Verify isolation ---
    loaded_a = await store.load_memory(
        tenant_id=tenant_a,
        session_id=session_id,
    )
    loaded_b = await store.load_memory(
        tenant_id=tenant_b,
        session_id=session_id,
    )

    assert [m.content for m in loaded_a.get_all()] == ["message-from-A"]
    assert [m.content for m in loaded_b.get_all()] == ["message-from-B"]

    # --- Delete A, B must remain ---
    await store.delete_session(
        tenant_id=tenant_a,
        session_id=session_id,
    )

    remaining_b = await store.load_memory(
        tenant_id=tenant_b,
        session_id=session_id,
    )

    assert [m.content for m in remaining_b.get_all()] == ["message-from-B"]