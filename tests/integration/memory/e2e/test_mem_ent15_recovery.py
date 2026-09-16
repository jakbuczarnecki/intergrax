# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15: partial failure, reconciliation, ambiguous provider retry."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.memory.contracts.memory_control import (
    MemoryControlBackendError,
    MemoryControlPartialLifecycleError,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    MemoryControlReconcileRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_lifecycle import MemoryReconciliationDisposition
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.user_profile_manager import UserProfileManager
from tests.integration.memory.e2e.harness import (
    build_in_memory_memory_harness_with_recovery_projection,
)
from tests.unit.memory.resilience.fault_injection import FailAfterCommitOnceUserProfileStore

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, pytest.mark.gate]


@pytest.mark.asyncio
async def test_partial_projection_failure_then_reconcile_repairs_recall() -> None:
    harness, recovery = build_in_memory_memory_harness_with_recovery_projection()
    identity = harness.identity()
    scope = harness.user_scope(identity)

    with pytest.raises(MemoryControlPartialLifecycleError):
        await harness.plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="canonical survives projection glitch"),
        )

    profile = await harness.canonical_profile()
    assert profile.memory_entries
    assert recovery.entries == {}

    outcome = await harness.plane.reconcile(
        harness.identity(),
        harness.user_scope(),
        MemoryControlReconcileRequest(),
    )
    assert outcome.reconciliation is not None
    assert outcome.reconciliation.disposition is MemoryReconciliationDisposition.REPAIRED
    assert recovery.entries

    recall = await harness.plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="canonical", top_k=5),
    )
    assert recall.items

    partial_events = [
        event
        for event in harness.observability.events
        if event.operation is MemoryDiagnosticOperation.REMEMBER
        and event.outcome is MemoryDiagnosticOutcome.PARTIAL
    ]
    assert partial_events


@pytest.mark.asyncio
async def test_ambiguous_commit_control_plane_recall_matches_canonical_state() -> None:
    tenant = "tenant-ent15-ambiguous"
    user = "user-ambiguous"
    store = FailAfterCommitOnceUserProfileStore()
    manager = UserProfileManager(store, tenant_id=tenant)
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )
    identity = RequestIdentity(
        tenant_id=tenant,
        user_id=user,
        principal_type=PrincipalType.USER,
        auth_subject=user,
    )
    scope = user_memory_scope(identity)
    request = MemoryControlRememberRequest(content="ambiguous-but-persisted")

    with pytest.raises(MemoryControlBackendError):
        await plane.remember(identity, scope, request)

    profile = await manager.get_profile(user)
    assert len(profile.memory_entries) == 1

    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=5))
    assert len(recall.items) == 1
    assert recall.items[0].content == "ambiguous-but-persisted"

    reconcile = await plane.reconcile(identity, scope, MemoryControlReconcileRequest())
    assert reconcile.reconciliation is not None
