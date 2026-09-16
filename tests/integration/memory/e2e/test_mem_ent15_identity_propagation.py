# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R: trusted RequestIdentity through lifecycle projection to entity indexer."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlRememberRequest,
    MemoryControlReconcileRequest,
)
from intergrax.memory.contracts.entity_temporal_memory import EntityMemoryScope
from intergrax.memory.contracts.memory_models import MemoryKind
from intergrax.memory.contracts.memory_recall import MemorySupersessionIntent
from tests.integration.memory.e2e.harness import build_in_memory_memory_harness

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


def _trusted_identity(harness) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=harness.tenant_id,
        user_id=harness.user_id,
        principal_type=PrincipalType.SERVICE,
        auth_subject="auth-subject-mem-ent15-r",
    )


@pytest.mark.asyncio
async def test_remember_propagates_trusted_identity_to_entity_indexer() -> None:
    harness = build_in_memory_memory_harness()
    identity = _trusted_identity(harness)
    scope = harness.user_scope(identity)
    assert harness.entity_indexer is not None

    result = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="identity fidelity fact", kind=MemoryKind.USER_FACT),
    )
    assert result.entry_id
    observed = harness.entity_indexer.observed_indexing[-1]
    assert observed == identity


@pytest.mark.asyncio
async def test_supersession_propagates_trusted_identity_on_projection_update() -> None:
    harness = build_in_memory_memory_harness(tenant_id="tenant-id-super", user_id="user-id-super")
    identity = _trusted_identity(harness)
    scope = harness.user_scope(identity)
    assert harness.entity_indexer is not None

    older = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="older fact", kind=MemoryKind.USER_FACT),
    )
    newer = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="newer fact", kind=MemoryKind.USER_FACT),
    )
    assert older.entry_id and newer.entry_id
    before = len(harness.entity_indexer.observed_indexing)

    await harness.plane.apply_memory_supersession(
        identity,
        scope,
        MemorySupersessionIntent(
            superseded_memory_id=older.entry_id,
            superseding_memory_id=newer.entry_id,
            reason="mem-ent15-r-supersession",
        ),
    )
    assert len(harness.entity_indexer.observed_indexing) > before
    assert harness.entity_indexer.observed_indexing[-1] == identity


@pytest.mark.asyncio
async def test_reconcile_propagates_trusted_identity() -> None:
    harness = build_in_memory_memory_harness(tenant_id="tenant-id-recon", user_id="user-id-recon")
    identity = _trusted_identity(harness)
    scope = harness.user_scope(identity)
    assert harness.entity_indexer is not None

    remembered = await harness.plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="reconcile identity", kind=MemoryKind.USER_FACT),
    )
    assert remembered.entry_id
    entity_scope = EntityMemoryScope(tenant_id=harness.tenant_id, user_id=harness.user_id)
    assert harness.entity_store is not None
    harness.entity_store.delete_by_source_memory(entity_scope, remembered.entry_id)
    harness.entity_indexer.observed_indexing.clear()

    await harness.plane.reconcile(identity, scope, MemoryControlReconcileRequest())
    assert harness.entity_indexer.observed_indexing
    assert harness.entity_indexer.observed_indexing[-1] == identity


@pytest.mark.asyncio
async def test_tenant_mismatch_identity_scope_denied_before_projection() -> None:
    harness = build_in_memory_memory_harness()
    identity = RequestIdentity(
        tenant_id="tenant-a",
        user_id=harness.user_id,
        principal_type=PrincipalType.USER,
        auth_subject="subject-a",
    )
    scope = harness.user_scope(
        RequestIdentity(
            tenant_id=harness.tenant_id,
            user_id=harness.user_id,
            principal_type=PrincipalType.USER,
            auth_subject="subject-b",
        )
    )
    assert harness.entity_indexer is not None
    before = len(harness.entity_indexer.observed_indexing)

    with pytest.raises(MemoryControlAccessDenied):
        await harness.plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="must not index", kind=MemoryKind.USER_FACT),
        )
    assert len(harness.entity_indexer.observed_indexing) == before
