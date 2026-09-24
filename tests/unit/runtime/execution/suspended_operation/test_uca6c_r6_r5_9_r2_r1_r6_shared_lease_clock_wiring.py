# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R2-R1-R6 — shared lease clock wiring across suspended-work composition."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationClaimOutcome,
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.execution.execution_bound_catalog_tool_composition import (
    build_execution_bound_catalog_tool_composition,
)
from intergrax.runtime.execution.suspended_operation.claim_lifecycle_wiring import (
    claim_lifecycle_from_hitl_continuation,
)
from intergrax.runtime.execution.suspended_operation.composition import (
    wire_suspended_execution_operation_store,
)
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    DocumentStoreSuspendedExecutionOperationStore,
    reconnect_document_store_suspended_operation_store,
)
from intergrax.runtime.execution.suspended_operation.reentry_coordinator import (
    ExecutionSuspendedWorkReentryCoordinator,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.providers.sandbox.bundle import CODE_EXEC_TOOL_ID
from tests.unit.autonomous_work.uca6c_r5_r2_strict_fixtures import (
    uca6c_strict_r6_durable_wiring,
)
from tests.unit.runtime.execution.suspended_operation.test_suspended_operation_store import (
    _descriptor,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_r1_r5_lease_clock_authority_replacement import (
    ManualUtcClock,
    _PRODUCTION_CLOCK_SCOPE,
    _blocked_claimed,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_WIRING_PATH = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "suspended_operation"
    / "claim_lifecycle_wiring.py"
)


def _minimal_reentry(
    store: object,
    clock: ManualUtcClock,
) -> ExecutionSuspendedWorkReentryCoordinator:
    return ExecutionSuspendedWorkReentryCoordinator(
        store=store,
        continuation_port=MagicMock(),
        tool_registry=MagicMock(),
        catalog_host=MagicMock(),
        catalog_invoker=MagicMock(),
        codec_registry=MagicMock(),
        binding_resolver=MagicMock(),
        claim_owner_id="host-b",
        utc_clock=clock,
    )


def _lifecycle_from_hitl(store: object, clock: ManualUtcClock) -> object:
    reentry = _minimal_reentry(store, clock)
    hitl = InternalOrchestrationContinuation(
        port=MagicMock(),
        lifecycle_driver=MagicMock(),
        suspended_work_reentry_coordinator=reentry,
    )
    lifecycle = claim_lifecycle_from_hitl_continuation(hitl)
    assert lifecycle is not None
    return lifecycle


def _blocked_claimed_on_store(
    store: object,
    clock: ManualUtcClock,
    *,
    lease_expires_at: datetime,
) -> object:
    descriptor = _descriptor()
    store.prepare(descriptor)
    pending = PendingExecutionContinuation(
        continuation_id=descriptor.continuation_id,
        identity=descriptor.identity,
        lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        reason=ContinuationReason.COMPLIANCE,
        revision=1,
        governed_correlation=GovernedContinuationCorrelation(
            continuation_request_id=descriptor.continuation_id,
            reason=ContinuationReason.COMPLIANCE,
            task_id=descriptor.identity.task_id,
            run_id=descriptor.identity.run_id,
            attempt_id=descriptor.identity.attempt_id,
            execution_id=descriptor.identity.execution_id,
            operation_id=descriptor.invocation_scope_id,
        ),
        pause_id="pause_1",
        human_request_id="hr_1",
    )
    blocked = store.block(
        suspended_operation_id=descriptor.suspended_operation_id,
        expected_materialization_revision=0,
        continuation=pending,
        governed_correlation=pending.governed_correlation,
    )
    assert blocked.descriptor is not None
    claimed = store.claim(
        suspended_operation_id=descriptor.suspended_operation_id,
        expected_materialization_revision=blocked.descriptor.materialization_revision,
        owner_id="host-a",
        lease_expires_at=lease_expires_at,
    )
    assert claimed.outcome is SuspendedOperationClaimOutcome.CLAIMED
    assert claimed.descriptor is not None
    return claimed.descriptor


def test_canonical_composition_shares_single_manual_clock(tmp_path: Path) -> None:
    from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
    from intergrax.runtime.sandbox.isolation_gate import SandboxIsolationAvailability
    from intergrax.tools.registry.runtime import ToolRegistry

    clock = ManualUtcClock(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    bundle = uca6c_strict_r6_durable_wiring(tmp_path)
    sandbox_availability = lambda: SandboxIsolationAvailability(  # noqa: E731
        session_configured=True,
        host_configured=True,
        healthy=True,
    )
    composition = build_execution_bound_catalog_tool_composition(
        registry=ToolRegistry(),
        policy_bundle=RuntimePolicyBundle(),
        caller_agent_id="worker-clock-r6",
        sandbox_availability=sandbox_availability,
        production_mode=False,
        scope_policy=StaticToolScopePolicy(allowed_tools={CODE_EXEC_TOOL_ID}),
        agent_runtime_governance=None,
        canonical_inner_execution_guard=None,
        meaningful_side_effect_authorization=None,
        document_store=bundle["document_store"],
        continuation_dependencies=bundle["continuation_dependencies"],
        reentry_claim_owner_id="host-a",
        durable_wiring_binding_resolver=bundle.get(
            "durable_wiring_binding_resolver",
        ),
        task_checkpoint_store=bundle["task_checkpoint_store"],
        utc_clock=clock,
    )
    reentry = composition.suspended_work_reentry_coordinator
    assert reentry is not None
    assert reentry.utc_clock is clock
    deps = bundle["continuation_dependencies"]
    hitl = InternalOrchestrationContinuation(
        port=deps.continuation,
        lifecycle_driver=deps.lifecycle_driver,
        suspended_work_reentry_coordinator=reentry,
    )
    lifecycle = claim_lifecycle_from_hitl_continuation(hitl)
    assert lifecycle is not None
    assert lifecycle.utc_clock is clock


def test_wired_in_memory_store_and_lifecycle_share_clock_behavior() -> None:
    lease_end = datetime(2026, 4, 1, 12, 1, tzinfo=timezone.utc)
    clock = ManualUtcClock(datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc))
    store = wire_suspended_execution_operation_store(utc_clock=clock)
    lifecycle = _lifecycle_from_hitl(store, clock)
    assert lifecycle.utc_clock is clock
    descriptor = _blocked_claimed_on_store(
        store,
        clock,
        lease_expires_at=lease_end,
    )
    assert descriptor.claim_ownership is not None
    assert (
        lifecycle.reclaim_expired_lease(
            descriptor,
            lease_expires_at=lease_end + timedelta(minutes=5),
            expected_fence=descriptor.claim_ownership.fence,
        )
        is None
    )
    denied = store.reclaim(
        suspended_operation_id=descriptor.suspended_operation_id,
        expected_materialization_revision=descriptor.materialization_revision,
        owner_id="host-b",
        lease_expires_at=lease_end + timedelta(minutes=5),
        expected_fence=descriptor.claim_ownership.fence,
    )
    assert denied.outcome is SuspendedOperationMutationOutcome.INVALID_STATE


def test_shared_clock_exact_expiry_boundary() -> None:
    lease_end = datetime(2026, 4, 1, 12, 1, tzinfo=timezone.utc)
    clock = ManualUtcClock(datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc))
    store, lifecycle, descriptor = _blocked_claimed(
        clock,
        lease_expires_at=lease_end,
    )
    wired = _lifecycle_from_hitl(store, clock)
    assert wired.utc_clock is clock
    assert descriptor.claim_ownership is not None
    clock.set_now(lease_end)
    assert (
        lifecycle.reclaim_expired_lease(
            descriptor,
            lease_expires_at=lease_end + timedelta(minutes=5),
            expected_fence=descriptor.claim_ownership.fence,
        )
        is not None
    )
    assert (
        wired.reclaim_expired_lease(
            descriptor,
            lease_expires_at=lease_end + timedelta(minutes=5),
            expected_fence=descriptor.claim_ownership.fence,
        )
        is None
    )


def test_shared_clock_after_expiry_reclaim() -> None:
    lease_end = datetime(2026, 4, 1, 12, 1, tzinfo=timezone.utc)
    clock = ManualUtcClock(datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc))
    store, _, descriptor = _blocked_claimed(clock, lease_expires_at=lease_end)
    wired = _lifecycle_from_hitl(store, clock)
    assert descriptor.claim_ownership is not None
    clock.set_now(lease_end + timedelta(minutes=1))
    context = wired.reclaim_expired_lease(
        descriptor,
        lease_expires_at=lease_end + timedelta(minutes=10),
        expected_fence=descriptor.claim_ownership.fence,
    )
    assert context is not None
    assert context.claim_authority.owner_id == "host-b"


def test_document_store_variant_uses_injected_shared_clock() -> None:
    lease_end = datetime(2026, 5, 1, 12, 1, tzinfo=timezone.utc)
    clock = ManualUtcClock(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc))
    document_store = InMemoryDocumentStore()
    store = DocumentStoreSuspendedExecutionOperationStore(
        document_store,
        utc_clock=clock,
    )
    lifecycle = _lifecycle_from_hitl(store, clock)
    descriptor = _blocked_claimed_on_store(
        store,
        clock,
        lease_expires_at=lease_end,
    )
    assert descriptor.claim_ownership is not None
    clock.set_now(lease_end)
    assert lifecycle.reclaim_expired_lease(
        descriptor,
        lease_expires_at=lease_end + timedelta(minutes=5),
        expected_fence=descriptor.claim_ownership.fence,
    )


def test_reconnect_preserves_injected_clock_instance() -> None:
    lease_end = datetime(2026, 7, 1, 12, 1, tzinfo=timezone.utc)
    clock = ManualUtcClock(datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc))
    document_store = InMemoryDocumentStore()
    store_a = DocumentStoreSuspendedExecutionOperationStore(
        document_store,
        utc_clock=clock,
    )
    descriptor = _blocked_claimed_on_store(
        store_a,
        clock,
        lease_expires_at=lease_end,
    )
    store_b = reconnect_document_store_suspended_operation_store(
        document_store,
        utc_clock=clock,
    )
    lifecycle_b = _lifecycle_from_hitl(store_b, clock)
    assert lifecycle_b.utc_clock is clock
    assert descriptor.claim_ownership is not None
    clock.set_now(lease_end + timedelta(seconds=1))
    context = lifecycle_b.reclaim_expired_lease(
        descriptor,
        lease_expires_at=lease_end + timedelta(minutes=5),
        expected_fence=descriptor.claim_ownership.fence,
    )
    assert context is not None


def test_claim_lifecycle_wiring_uses_reentry_public_clock() -> None:
    source = _WIRING_PATH.read_text(encoding="utf-8")
    assert "utc_clock=reentry.utc_clock" in source
    assert "store._utc_clock" not in source
    assert "store._backing" not in source


def test_reentry_coordinator_lease_uses_injected_clock_not_wall_clock() -> None:
    reentry_path = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "execution"
        / "suspended_operation"
        / "reentry_coordinator.py"
    )
    source = reentry_path.read_text(encoding="utf-8")
    assert "datetime.now(timezone.utc)" not in source
    assert "self.utc_clock.now_utc()" in source


def test_no_new_private_cross_component_clock_access_in_production_scope() -> None:
    forbidden_patterns = (
        "store._utc_clock",
        "store._backing._utc_clock",
    )
    for path in _PRODUCTION_CLOCK_SCOPE.rglob("*.py"):
        if path.name in {
            "store_engine.py",
            "in_memory_store.py",
            "document_store_suspended_operation_store.py",
        }:
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern not in source, f"{path.relative_to(_REPO_ROOT)}: {pattern}"
