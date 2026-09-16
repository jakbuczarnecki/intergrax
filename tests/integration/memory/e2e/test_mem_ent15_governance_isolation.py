# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15: governance ALLOW/DENY and tenant isolation."""

from __future__ import annotations

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlGovernanceDenied,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
)
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
)
from tests.integration.memory.e2e.harness import build_in_memory_memory_harness
from tests.unit.memory.test_mem_ent10b_specialized_mutation_governance import _deny_governance

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
async def test_governance_allow_persists_canonical_and_projection() -> None:
    harness = build_in_memory_memory_harness(tenant_id="tenant-ent15-allow", user_id="user-allow")
    identity = harness.identity()
    scope = harness.user_scope(identity)
    result = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="allowed fact"),
    )
    profile = await harness.canonical_profile()
    assert profile.memory_entries
    assert harness.entity_record_for_entry(harness.user_id, result.entry_id or "") is not None


@pytest.mark.asyncio
async def test_governance_deny_zero_canonical_and_projection_writes() -> None:
    harness = build_in_memory_memory_harness(
        tenant_id="tenant-ent15-deny",
        user_id="user-deny",
        governance=_deny_governance(),
    )
    identity = harness.identity()
    scope = harness.user_scope(identity)
    with pytest.raises(MemoryControlGovernanceDenied):
        await harness.plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="must not persist"),
        )
    profile = await harness.canonical_profile()
    assert profile.memory_entries == []
    denied = [
        event
        for event in harness.observability.events
        if event.operation is MemoryDiagnosticOperation.REMEMBER
        and event.outcome is MemoryDiagnosticOutcome.DENIED
    ]
    assert denied


@pytest.mark.asyncio
async def test_cross_tenant_recall_isolation() -> None:
    tenant_a = "tenant-ent15-a"
    tenant_b = "tenant-ent15-b"
    user = "user-shared"

    harness_a = build_in_memory_memory_harness(tenant_id=tenant_a, user_id=user)
    harness_b = build_in_memory_memory_harness(tenant_id=tenant_b, user_id=user)
    id_a = harness_a.identity(user_id=user, tenant_id=tenant_a)
    id_b = harness_b.identity(user_id=user, tenant_id=tenant_b)
    scope_a = harness_a.user_scope(id_a)
    scope_b = harness_b.user_scope(id_b)

    await harness_a.plane.remember(
        id_a, scope_a, MemoryControlRememberRequest(content="secret tenant A marker")
    )
    await harness_b.plane.remember(
        id_b, scope_b, MemoryControlRememberRequest(content="secret tenant B marker")
    )

    recall_a = await harness_a.plane.recall(
        id_a, scope_a, MemoryControlRecallRequest(query="tenant", top_k=5)
    )
    recall_b = await harness_b.plane.recall(
        id_b, scope_b, MemoryControlRecallRequest(query="tenant", top_k=5)
    )
    texts_a = {item.content for item in recall_a.items}
    texts_b = {item.content for item in recall_b.items}
    assert "secret tenant A marker" in texts_a
    assert "secret tenant B marker" not in texts_a
    assert "secret tenant B marker" in texts_b
    assert "secret tenant A marker" not in texts_b
