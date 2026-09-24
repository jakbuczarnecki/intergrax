# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R2-R1-R5 — lease clock contract and authority replacement hardening."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationClaimOutcome,
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.resume_authority_context import (
    ExecutionSuspendedWorkResumeAuthorityContext,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.runtime.execution.deadline_authority.system_clocks import SystemUtcClock
from intergrax.runtime.execution.suspended_operation.claim_lifecycle import (
    ExecutionSuspendedWorkClaimLifecycleCoordinator,
)
from intergrax.runtime.execution.suspended_operation.in_memory_store import (
    InMemorySuspendedExecutionOperationStore,
)
from intergrax.runtime.execution.suspended_operation.resume_authority_transport import (
    ExecutionSuspendedWorkResumeAuthorityTransport,
    ExecutionSuspendedWorkResumeAuthorityTransportAuthorityConflictError,
    ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome,
)
from tests.unit.runtime.execution.suspended_operation.test_suspended_operation_store import (
    _descriptor,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PRODUCTION_CLOCK_SCOPE = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "suspended_operation"
)


class ManualUtcClock:
    def __init__(self, initial: datetime) -> None:
        if initial.tzinfo is None:
            raise ValueError("initial instant must be timezone-aware")
        self._now = initial

    def now_utc(self) -> datetime:
        return self._now

    def set_now(self, value: datetime) -> None:
        if value.tzinfo is None:
            raise ValueError("instant must be timezone-aware")
        self._now = value


def _authority(
    *, fence: int = 1, owner: str = "host-a"
) -> SuspendedOperationClaimAuthority:
    return SuspendedOperationClaimAuthority(
        owner_id=owner,
        fence=fence,
        materialization_revision=1,
        pause_generation=3,
    )


def _context(
    continuation_id: str,
    authority: SuspendedOperationClaimAuthority,
) -> ExecutionSuspendedWorkResumeAuthorityContext:
    return ExecutionSuspendedWorkResumeAuthorityContext(
        continuation_id=continuation_id,
        claim_authority=authority,
    )


def _blocked_claimed(
    clock: ManualUtcClock,
    *,
    lease_expires_at: datetime,
) -> tuple[
    InMemorySuspendedExecutionOperationStore,
    ExecutionSuspendedWorkClaimLifecycleCoordinator,
    object,
]:
    store = InMemorySuspendedExecutionOperationStore(utc_clock=clock)
    lifecycle = ExecutionSuspendedWorkClaimLifecycleCoordinator(
        store=store,
        claim_owner_id="host-b",
        utc_clock=clock,
    )
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
    return store, lifecycle, claimed.descriptor


def test_system_utc_clock_is_timezone_aware() -> None:
    instant = SystemUtcClock().now_utc()
    assert instant.tzinfo is not None
    assert instant.tzinfo == timezone.utc


def test_manual_utc_clock_is_deterministic() -> None:
    fixed = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    clock = ManualUtcClock(fixed)
    assert clock.now_utc() == fixed
    advanced = fixed + timedelta(hours=1)
    clock.set_now(advanced)
    assert clock.now_utc() == advanced


def test_reclaim_before_expiry_blocked_by_lifecycle_and_store() -> None:
    lease_end = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    clock = ManualUtcClock(lease_end - timedelta(minutes=1))
    store, lifecycle, descriptor = _blocked_claimed(
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


def test_reclaim_at_expiry_boundary_allowed() -> None:
    lease_end = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    clock = ManualUtcClock(lease_end - timedelta(hours=1))
    store, lifecycle, descriptor = _blocked_claimed(
        clock,
        lease_expires_at=lease_end,
    )
    assert descriptor.claim_ownership is not None
    clock.set_now(lease_end)
    context = lifecycle.reclaim_expired_lease(
        descriptor,
        lease_expires_at=lease_end + timedelta(minutes=5),
        expected_fence=descriptor.claim_ownership.fence,
    )
    assert context is not None
    assert context.claim_authority.owner_id == "host-b"
    reloaded = store.load(descriptor.suspended_operation_id)
    assert reloaded is not None
    assert reloaded.claim_ownership is not None
    assert reloaded.claim_ownership.owner_id == "host-b"


def test_reclaim_after_expiry_cross_host() -> None:
    lease_end = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
    clock = ManualUtcClock(lease_end - timedelta(hours=1))
    store, lifecycle, descriptor = _blocked_claimed(
        clock,
        lease_expires_at=lease_end,
    )
    assert descriptor.claim_ownership is not None
    clock.set_now(lease_end + timedelta(seconds=1))
    context = lifecycle.reclaim_expired_lease(
        descriptor,
        lease_expires_at=lease_end + timedelta(minutes=5),
        expected_fence=descriptor.claim_ownership.fence,
    )
    assert context is not None
    assert context.claim_authority.owner_id == "host-b"
    assert context.claim_authority.fence > descriptor.claim_ownership.fence


def test_no_private_store_engine_utc_now_in_production_scope() -> None:
    forbidden = "store_engine._utc_now"
    for path in _PRODUCTION_CLOCK_SCOPE.rglob("*.py"):
        if path.name == "store_engine.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert forbidden not in source, path.relative_to(_REPO_ROOT)


def test_transport_empty_deliver() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    ctx = _context("c1", _authority())
    transport.deliver(ctx)
    assert transport.peek() == ctx


def test_transport_idempotent_deliver() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    ctx = _context("c1", _authority(fence=2))
    transport.deliver(ctx)
    transport.deliver(ctx)
    assert transport.peek() == ctx


def test_transport_different_authority_generic_deliver_rejected() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    transport.deliver(_context("c1", _authority(fence=1)))
    with pytest.raises(
        ExecutionSuspendedWorkResumeAuthorityTransportAuthorityConflictError
    ):
        transport.deliver(_context("c1", _authority(fence=2)))


def test_transport_explicit_replace_applied() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    old = _authority(fence=1)
    new = _authority(fence=2)
    transport.deliver(_context("c1", old))
    outcome = transport.replace_for_continuation(
        continuation_id="c1",
        expected_authority=old,
        replacement=_context("c1", new),
    )
    assert (
        outcome
        is ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.APPLIED
    )
    assert transport.peek() == _context("c1", new)


def test_transport_explicit_replace_stale_expected() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    current = _authority(fence=2, owner="host-b")
    transport.deliver(_context("c1", current))
    outcome = transport.replace_for_continuation(
        continuation_id="c1",
        expected_authority=_authority(fence=1),
        replacement=_context("c1", _authority(fence=3)),
    )
    assert outcome is (
        ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.STALE_AUTHORITY
    )
    assert transport.peek() == _context("c1", current)


def test_transport_replace_wrong_continuation() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    old = _authority(fence=1)
    transport.deliver(_context("c1", old))
    outcome = transport.replace_for_continuation(
        continuation_id="c2",
        expected_authority=old,
        replacement=_context("c2", _authority(fence=2)),
    )
    assert outcome is (
        ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.CONTINUATION_MISMATCH
    )


def test_transport_replace_idempotent() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    old = _authority(fence=1)
    new = _authority(fence=2)
    transport.deliver(_context("c1", old))
    transport.replace_for_continuation(
        continuation_id="c1",
        expected_authority=old,
        replacement=_context("c1", new),
    )
    outcome = transport.replace_for_continuation(
        continuation_id="c1",
        expected_authority=old,
        replacement=_context("c1", new),
    )
    assert (
        outcome
        is ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.IDEMPOTENT
    )
