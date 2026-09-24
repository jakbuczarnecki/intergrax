# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R2 — multi-host claim / fence / reclaim closure."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationClaimOutcome,
    SuspendedOperationClaimResult,
    SuspendedOperationMutationOutcome,
    SuspendedOperationMutationResult,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.persistence_conflict import (
    SuspendedOperationPersistenceConflictError,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.execution.suspended_operation import store_engine
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    DocumentStoreSuspendedExecutionOperationStore,
    reconnect_document_store_suspended_operation_store,
)
from tests.unit.runtime.execution.suspended_operation.test_suspended_operation_store import (
    _descriptor,
)

pytestmark = pytest.mark.unit

OWNER_HOST_A = "uca6c-r59r2-host-a"
OWNER_HOST_B = "uca6c-r59r2-host-b"


@dataclass
class ClaimFenceCounters:
    claim_attempts: int = 0
    claim_successes: int = 0
    reclaims: int = 0
    stale_reblock_attempts: int = 0
    stale_consume_attempts: int = 0
    valid_consume_attempts: int = 0
    backend_physical_attempts: int = 0
    backend_logical_effects: int = 0


@dataclass
class MultiHostFixture:
    document_store: InMemoryDocumentStore
    store_a: DocumentStoreSuspendedExecutionOperationStore
    store_b: DocumentStoreSuspendedExecutionOperationStore
    descriptor: SuspendedExecutionOperationDescriptor
    blocked_revision: int
    counters: ClaimFenceCounters = field(default_factory=ClaimFenceCounters)


def build_shared_backend() -> InMemoryDocumentStore:
    return InMemoryDocumentStore()


def build_host_a(
    document_store: InMemoryDocumentStore,
) -> DocumentStoreSuspendedExecutionOperationStore:
    return DocumentStoreSuspendedExecutionOperationStore(document_store)


def build_host_b(
    document_store: InMemoryDocumentStore,
) -> DocumentStoreSuspendedExecutionOperationStore:
    return DocumentStoreSuspendedExecutionOperationStore(document_store)


def refresh_host_b(fixture: MultiHostFixture) -> None:
    """New store wrapper — same durable backend, fresh in-memory view."""
    fixture.store_b = reconnect_document_store_suspended_operation_store(
        fixture.document_store,
    )


def refresh_host_a(fixture: MultiHostFixture) -> None:
    fixture.store_a = reconnect_document_store_suspended_operation_store(
        fixture.document_store,
    )


def _pending_for(
    descriptor: SuspendedExecutionOperationDescriptor,
) -> PendingExecutionContinuation:
    return PendingExecutionContinuation(
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


def _block_on_host(
    store: DocumentStoreSuspendedExecutionOperationStore,
    descriptor: SuspendedExecutionOperationDescriptor,
) -> int:
    pending = _pending_for(descriptor)
    correlation = pending.governed_correlation
    assert correlation is not None
    store.prepare(descriptor)
    blocked = store.block(
        suspended_operation_id=descriptor.suspended_operation_id,
        expected_materialization_revision=0,
        continuation=pending,
        governed_correlation=correlation,
    )
    assert blocked.outcome is SuspendedOperationMutationOutcome.APPLIED
    assert blocked.descriptor is not None
    return blocked.descriptor.materialization_revision


def build_blocked_multi_host() -> MultiHostFixture:
    backend = build_shared_backend()
    store_a = build_host_a(backend)
    store_b = build_host_b(backend)
    descriptor = _descriptor()
    blocked_revision = _block_on_host(store_a, descriptor)
    return MultiHostFixture(
        document_store=backend,
        store_a=store_a,
        store_b=store_b,
        descriptor=descriptor,
        blocked_revision=blocked_revision,
    )


def _lease_from_now(minutes: int = 5) -> datetime:
    return datetime.now(UTC) + timedelta(minutes=minutes)


def claim_as(
    fixture: MultiHostFixture,
    host: str,
    *,
    expected_revision: int,
    owner_id: str,
) -> SuspendedOperationClaimResult:
    fixture.counters.claim_attempts += 1
    store = fixture.store_a if host == "a" else fixture.store_b
    try:
        result = store.claim(
            suspended_operation_id=fixture.descriptor.suspended_operation_id,
            expected_materialization_revision=expected_revision,
            owner_id=owner_id,
            lease_expires_at=_lease_from_now(),
        )
    except SuspendedOperationPersistenceConflictError:
        reloaded = store.load(fixture.descriptor.suspended_operation_id)
        assert reloaded is not None
        result = store.claim(
            suspended_operation_id=fixture.descriptor.suspended_operation_id,
            expected_materialization_revision=reloaded.materialization_revision,
            owner_id=owner_id,
            lease_expires_at=_lease_from_now(),
        )
    if result.outcome is SuspendedOperationClaimOutcome.CLAIMED:
        fixture.counters.claim_successes += 1
    return result


@contextmanager
def advance_lease_clock(target: datetime) -> Iterator[None]:
    with patch.object(store_engine, "_utc_now", return_value=target):
        yield


def reclaim_as(
    fixture: MultiHostFixture,
    host: str,
    *,
    expected_revision: int,
    owner_id: str,
    expected_fence: int,
    lease_at: datetime,
) -> SuspendedOperationMutationResult:
    store = fixture.store_a if host == "a" else fixture.store_b
    try:
        result = store.reclaim(
            suspended_operation_id=fixture.descriptor.suspended_operation_id,
            expected_materialization_revision=expected_revision,
            owner_id=owner_id,
            lease_expires_at=lease_at,
            expected_fence=expected_fence,
        )
    except SuspendedOperationPersistenceConflictError:
        reloaded = store.load(fixture.descriptor.suspended_operation_id)
        assert reloaded is not None
        assert reloaded.claim_ownership is not None
        result = store.reclaim(
            suspended_operation_id=fixture.descriptor.suspended_operation_id,
            expected_materialization_revision=reloaded.materialization_revision,
            owner_id=owner_id,
            lease_expires_at=lease_at,
            expected_fence=reloaded.claim_ownership.fence,
        )
    if result.outcome is SuspendedOperationMutationOutcome.APPLIED:
        fixture.counters.reclaims += 1
    return result


def test_simultaneous_claim_race_exactly_one_winner() -> None:
    fixture = build_blocked_multi_host()
    barrier = threading.Barrier(2)
    results: list[tuple[str, SuspendedOperationClaimResult]] = []
    errors: list[BaseException] = []

    def _race(
        host: str, owner: str, store: DocumentStoreSuspendedExecutionOperationStore
    ) -> None:
        try:
            barrier.wait()
            fixture.counters.claim_attempts += 1
            try:
                outcome = store.claim(
                    suspended_operation_id=fixture.descriptor.suspended_operation_id,
                    expected_materialization_revision=fixture.blocked_revision,
                    owner_id=owner,
                    lease_expires_at=_lease_from_now(),
                )
            except SuspendedOperationPersistenceConflictError:
                reloaded = store.load(fixture.descriptor.suspended_operation_id)
                assert reloaded is not None
                outcome = store.claim(
                    suspended_operation_id=fixture.descriptor.suspended_operation_id,
                    expected_materialization_revision=reloaded.materialization_revision,
                    owner_id=owner,
                    lease_expires_at=_lease_from_now(),
                )
            if outcome.outcome is SuspendedOperationClaimOutcome.CLAIMED:
                fixture.counters.claim_successes += 1
            results.append((host, outcome))
        except BaseException as exc:
            errors.append(exc)

    t_a = threading.Thread(
        target=_race,
        args=("a", OWNER_HOST_A, fixture.store_a),
    )
    t_b = threading.Thread(
        target=_race,
        args=("b", OWNER_HOST_B, fixture.store_b),
    )
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()
    assert not errors
    winners = [
        item
        for item in results
        if item[1].outcome is SuspendedOperationClaimOutcome.CLAIMED
    ]
    losers = [
        item
        for item in results
        if item[1].outcome is not SuspendedOperationClaimOutcome.CLAIMED
    ]
    assert len(winners) == 1
    assert len(losers) == 1
    assert fixture.counters.claim_successes == 1
    winner_host, winner = winners[0]
    loser_host, loser = losers[0]
    assert winner.descriptor is not None
    assert winner.descriptor.claim_ownership is not None
    assert loser.outcome in {
        SuspendedOperationClaimOutcome.STALE_REVISION,
        SuspendedOperationClaimOutcome.ALREADY_CLAIMED,
    }
    active = fixture.store_b.load(fixture.descriptor.suspended_operation_id)
    assert active is not None
    assert (
        active.materialization_state is SuspendedOperationMaterializationState.CLAIMED
    )
    assert active.claim_ownership is not None
    assert active.claim_ownership.owner_id == winner.descriptor.claim_ownership.owner_id
    assert {winner_host, loser_host} == {"a", "b"}


def test_reclaim_before_lease_expiry_fail_closed() -> None:
    fixture = build_blocked_multi_host()
    claimed = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed.outcome is SuspendedOperationClaimOutcome.CLAIMED
    assert claimed.descriptor is not None
    assert claimed.descriptor.claim_ownership is not None
    fence = claimed.descriptor.claim_ownership.fence
    early = datetime.now(UTC) + timedelta(minutes=1)
    with advance_lease_clock(early):
        denied = reclaim_as(
            fixture,
            "b",
            expected_revision=claimed.descriptor.materialization_revision,
            owner_id=OWNER_HOST_B,
            expected_fence=fence,
            lease_at=early + timedelta(minutes=5),
        )
    assert denied.outcome is SuspendedOperationMutationOutcome.INVALID_STATE


def test_reclaim_after_expiry_new_fence_and_owner() -> None:
    fixture = build_blocked_multi_host()
    claimed_a = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed_a.descriptor is not None
    assert claimed_a.descriptor.claim_ownership is not None
    old_fence = claimed_a.descriptor.claim_ownership.fence
    pause_before = claimed_a.descriptor.pause_generation
    identity_before = claimed_a.descriptor.identity
    expired_now = datetime.now(UTC) + timedelta(hours=2)
    with advance_lease_clock(expired_now):
        reclaimed = reclaim_as(
            fixture,
            "b",
            expected_revision=claimed_a.descriptor.materialization_revision,
            owner_id=OWNER_HOST_B,
            expected_fence=old_fence,
            lease_at=expired_now + timedelta(minutes=5),
        )
    assert reclaimed.outcome is SuspendedOperationMutationOutcome.APPLIED
    assert reclaimed.descriptor is not None
    assert reclaimed.descriptor.claim_ownership is not None
    assert reclaimed.descriptor.claim_ownership.owner_id == OWNER_HOST_B
    assert reclaimed.descriptor.claim_ownership.fence > old_fence
    assert reclaimed.descriptor.pause_generation == pause_before
    assert reclaimed.descriptor.identity == identity_before
    assert (
        reclaimed.descriptor.suspended_operation_id
        == fixture.descriptor.suspended_operation_id
    )


def test_stale_host_cannot_consume_or_reblock_after_reclaim() -> None:
    fixture = build_blocked_multi_host()
    claimed_a = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed_a.descriptor is not None
    assert claimed_a.descriptor.claim_ownership is not None
    stale_fence = claimed_a.descriptor.claim_ownership.fence
    stale_revision = claimed_a.descriptor.materialization_revision
    expired_now = datetime.now(UTC) + timedelta(hours=2)
    with advance_lease_clock(expired_now):
        reclaimed = reclaim_as(
            fixture,
            "b",
            expected_revision=stale_revision,
            owner_id=OWNER_HOST_B,
            expected_fence=stale_fence,
            lease_at=expired_now + timedelta(minutes=5),
        )
    assert reclaimed.outcome is SuspendedOperationMutationOutcome.APPLIED
    assert reclaimed.descriptor is not None
    refresh_host_a(fixture)
    fixture.counters.stale_consume_attempts += 1
    stale_consume = fixture.store_a.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=reclaimed.descriptor.materialization_revision,
        owner_id=OWNER_HOST_A,
        fence=stale_fence,
    )
    assert stale_consume.outcome is SuspendedOperationMutationOutcome.STALE_CLAIM
    next_cont = PendingExecutionContinuation(
        continuation_id="gcr_gen2",
        identity=fixture.descriptor.identity,
        lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        reason=ContinuationReason.SECURITY,
        revision=1,
        governed_correlation=GovernedContinuationCorrelation(
            continuation_request_id="gcr_gen2",
            reason=ContinuationReason.SECURITY,
            task_id=fixture.descriptor.identity.task_id,
            run_id=fixture.descriptor.identity.run_id,
            attempt_id=fixture.descriptor.identity.attempt_id,
            execution_id=fixture.descriptor.identity.execution_id,
            operation_id="dhr_gen2",
        ),
        pause_id="pause_2",
        human_request_id="hr_2",
    )
    next_correlation = next_cont.governed_correlation
    assert next_correlation is not None
    fixture.counters.stale_reblock_attempts += 1
    stale_reblock = fixture.store_a.authority_reblock_from_claimed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=reclaimed.descriptor.materialization_revision,
        expected_pause_generation=reclaimed.descriptor.pause_generation,
        expected_owner_id=OWNER_HOST_A,
        expected_fence=stale_fence,
        next_pause_generation=reclaimed.descriptor.pause_generation + 1,
        next_continuation=next_cont,
        next_governed_correlation=next_correlation,
        next_invocation_scope_id="dhr_gen2",
        next_authority_scope=SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE,
    )
    assert stale_reblock.outcome is SuspendedOperationMutationOutcome.STALE_CLAIM


def test_current_host_b_may_reblock_and_consume() -> None:
    fixture = build_blocked_multi_host()
    claimed = claim_as(
        fixture,
        "b",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_B,
    )
    assert claimed.descriptor is not None
    assert claimed.descriptor.claim_ownership is not None
    next_cont = PendingExecutionContinuation(
        continuation_id="gcr_gen2",
        identity=fixture.descriptor.identity,
        lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        reason=ContinuationReason.SECURITY,
        revision=1,
        governed_correlation=GovernedContinuationCorrelation(
            continuation_request_id="gcr_gen2",
            reason=ContinuationReason.SECURITY,
            task_id=fixture.descriptor.identity.task_id,
            run_id=fixture.descriptor.identity.run_id,
            attempt_id=fixture.descriptor.identity.attempt_id,
            execution_id=fixture.descriptor.identity.execution_id,
            operation_id="dhr_gen2",
        ),
        pause_id="pause_2",
        human_request_id="hr_2",
    )
    next_correlation = next_cont.governed_correlation
    assert next_correlation is not None
    reblocked = fixture.store_b.authority_reblock_from_claimed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=claimed.descriptor.materialization_revision,
        expected_pause_generation=claimed.descriptor.pause_generation,
        expected_owner_id=OWNER_HOST_B,
        expected_fence=claimed.descriptor.claim_ownership.fence,
        next_pause_generation=claimed.descriptor.pause_generation + 1,
        next_continuation=next_cont,
        next_governed_correlation=next_correlation,
        next_invocation_scope_id="dhr_gen2",
        next_authority_scope=SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE,
    )
    assert reblocked.outcome is SuspendedOperationMutationOutcome.APPLIED
    assert reblocked.descriptor is not None
    reclaimed = claim_as(
        fixture,
        "b",
        expected_revision=reblocked.descriptor.materialization_revision,
        owner_id=OWNER_HOST_B,
    )
    assert reclaimed.outcome is SuspendedOperationClaimOutcome.CLAIMED
    assert reclaimed.descriptor is not None
    assert reclaimed.descriptor.claim_ownership is not None
    fixture.counters.valid_consume_attempts += 1
    fixture.counters.backend_physical_attempts += 1
    consumed = fixture.store_b.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=reclaimed.descriptor.materialization_revision,
        owner_id=OWNER_HOST_B,
        fence=reclaimed.descriptor.claim_ownership.fence,
    )
    if consumed.outcome is SuspendedOperationMutationOutcome.APPLIED:
        fixture.counters.backend_logical_effects += 1
    assert consumed.outcome is SuspendedOperationMutationOutcome.APPLIED


def test_stale_host_blocked_after_consumed() -> None:
    fixture = build_blocked_multi_host()
    claimed_b = claim_as(
        fixture,
        "b",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_B,
    )
    assert claimed_b.descriptor is not None
    assert claimed_b.descriptor.claim_ownership is not None
    consumed = fixture.store_b.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=claimed_b.descriptor.materialization_revision,
        owner_id=OWNER_HOST_B,
        fence=claimed_b.descriptor.claim_ownership.fence,
    )
    assert consumed.outcome is SuspendedOperationMutationOutcome.APPLIED
    assert consumed.descriptor is not None
    refresh_host_a(fixture)
    fixture.counters.stale_consume_attempts += 1
    stale = fixture.store_a.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=consumed.descriptor.materialization_revision,
        owner_id=OWNER_HOST_A,
        fence=claimed_b.descriptor.claim_ownership.fence,
    )
    assert stale.outcome in {
        SuspendedOperationMutationOutcome.STALE_CLAIM,
        SuspendedOperationMutationOutcome.INVALID_STATE,
        SuspendedOperationMutationOutcome.TERMINAL,
    }


def test_load_active_for_continuation_single_holder() -> None:
    fixture = build_blocked_multi_host()
    claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    refresh_host_b(fixture)
    from_a = fixture.store_a.load_active_for_continuation(
        fixture.descriptor.continuation_id
    )
    from_b = fixture.store_b.load_active_for_continuation(
        fixture.descriptor.continuation_id
    )
    assert from_a is not None
    assert from_b is not None
    assert from_a.suspended_operation_id == from_b.suspended_operation_id
    assert from_a.claim_ownership is not None
    assert from_a.claim_ownership.owner_id == OWNER_HOST_A


def test_stale_fence_with_valid_grant_metadata_still_blocked() -> None:
    """Approval/grant metadata does not substitute claim authority (Test K)."""
    fixture = build_blocked_multi_host()
    claimed_a = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed_a.descriptor is not None
    assert claimed_a.descriptor.claim_ownership is not None
    stale_fence = claimed_a.descriptor.claim_ownership.fence
    expired_now = datetime.now(UTC) + timedelta(hours=2)
    with advance_lease_clock(expired_now):
        reclaimed = reclaim_as(
            fixture,
            "b",
            expected_revision=claimed_a.descriptor.materialization_revision,
            owner_id=OWNER_HOST_B,
            expected_fence=stale_fence,
            lease_at=expired_now + timedelta(minutes=5),
        )
    assert reclaimed.descriptor is not None
    refresh_host_a(fixture)
    # Host A still holds stale fence as if a grant were cached locally.
    fixture.counters.stale_consume_attempts += 1
    blocked = fixture.store_a.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=reclaimed.descriptor.materialization_revision,
        owner_id=OWNER_HOST_A,
        fence=stale_fence,
        expected_pause_generation=claimed_a.descriptor.pause_generation,
    )
    assert blocked.outcome is SuspendedOperationMutationOutcome.STALE_CLAIM


def test_host_death_reclaim_without_manual_release() -> None:
    fixture = build_blocked_multi_host()
    claimed_a = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed_a.descriptor is not None
    assert claimed_a.descriptor.claim_ownership is not None
    expired_now = datetime.now(UTC) + timedelta(hours=3)
    with advance_lease_clock(expired_now):
        reclaimed = reclaim_as(
            fixture,
            "b",
            expected_revision=claimed_a.descriptor.materialization_revision,
            owner_id=OWNER_HOST_B,
            expected_fence=claimed_a.descriptor.claim_ownership.fence,
            lease_at=expired_now + timedelta(minutes=5),
        )
    assert reclaimed.outcome is SuspendedOperationMutationOutcome.APPLIED


def test_reclaim_preserves_execution_identity() -> None:
    fixture = build_blocked_multi_host()
    before = fixture.descriptor
    claimed_a = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed_a.descriptor is not None
    assert claimed_a.descriptor.claim_ownership is not None
    expired_now = datetime.now(UTC) + timedelta(hours=4)
    with advance_lease_clock(expired_now):
        reclaimed = reclaim_as(
            fixture,
            "b",
            expected_revision=claimed_a.descriptor.materialization_revision,
            owner_id=OWNER_HOST_B,
            expected_fence=claimed_a.descriptor.claim_ownership.fence,
            lease_at=expired_now + timedelta(minutes=5),
        )
    assert reclaimed.descriptor is not None
    after = reclaimed.descriptor
    assert after.identity.task_id == before.identity.task_id
    assert after.identity.run_id == before.identity.run_id
    assert after.identity.attempt_id == before.identity.attempt_id
    assert after.identity.execution_id == before.identity.execution_id
    assert after.suspended_operation_id == before.suspended_operation_id
    assert after.pause_generation == before.pause_generation


def test_dual_host_consume_exactly_one_logical_effect() -> None:
    fixture = build_blocked_multi_host()
    claimed_b = claim_as(
        fixture,
        "b",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_B,
    )
    assert claimed_b.descriptor is not None
    assert claimed_b.descriptor.claim_ownership is not None
    stale_fence = claimed_b.descriptor.claim_ownership.fence - 1
    fixture.counters.backend_physical_attempts += 2
    fixture.counters.stale_consume_attempts += 1
    fixture.counters.valid_consume_attempts += 1
    stale_try = fixture.store_a.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=claimed_b.descriptor.materialization_revision,
        owner_id=OWNER_HOST_A,
        fence=max(stale_fence, 0),
    )
    valid_try = fixture.store_b.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=claimed_b.descriptor.materialization_revision,
        owner_id=OWNER_HOST_B,
        fence=claimed_b.descriptor.claim_ownership.fence,
    )
    if valid_try.outcome is SuspendedOperationMutationOutcome.APPLIED:
        fixture.counters.backend_logical_effects += 1
    assert stale_try.outcome is SuspendedOperationMutationOutcome.STALE_CLAIM
    assert valid_try.outcome is SuspendedOperationMutationOutcome.APPLIED
    assert fixture.counters.backend_logical_effects == 1


def test_negative_unknown_operation() -> None:
    backend = build_shared_backend()
    store = build_host_a(backend)
    missing = store.claim(
        suspended_operation_id="suspended_operation_missing",
        expected_materialization_revision=0,
        owner_id=OWNER_HOST_A,
        lease_expires_at=_lease_from_now(),
    )
    assert missing.outcome is SuspendedOperationClaimOutcome.NOT_FOUND


def test_negative_consumed_cannot_reclaim() -> None:
    fixture = build_blocked_multi_host()
    claimed = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed.descriptor is not None
    assert claimed.descriptor.claim_ownership is not None
    consumed = fixture.store_a.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=claimed.descriptor.materialization_revision,
        owner_id=OWNER_HOST_A,
        fence=claimed.descriptor.claim_ownership.fence,
    )
    assert consumed.outcome is SuspendedOperationMutationOutcome.APPLIED
    assert consumed.descriptor is not None
    expired_now = datetime.now(UTC) + timedelta(hours=5)
    with advance_lease_clock(expired_now):
        denied = reclaim_as(
            fixture,
            "b",
            expected_revision=consumed.descriptor.materialization_revision,
            owner_id=OWNER_HOST_B,
            expected_fence=1,
            lease_at=expired_now + timedelta(minutes=5),
        )
    assert denied.outcome in {
        SuspendedOperationMutationOutcome.INVALID_STATE,
        SuspendedOperationMutationOutcome.TERMINAL,
    }


def test_negative_wrong_fence_same_revision() -> None:
    fixture = build_blocked_multi_host()
    claimed = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed.descriptor is not None
    assert claimed.descriptor.claim_ownership is not None
    ownership = claimed.descriptor.claim_ownership
    wrong = fixture.store_a.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=claimed.descriptor.materialization_revision,
        owner_id=OWNER_HOST_A,
        fence=ownership.fence + 99,
    )
    assert wrong.outcome is SuspendedOperationMutationOutcome.STALE_CLAIM


def test_negative_wrong_owner_valid_fence() -> None:
    fixture = build_blocked_multi_host()
    claimed = claim_as(
        fixture,
        "a",
        expected_revision=fixture.blocked_revision,
        owner_id=OWNER_HOST_A,
    )
    assert claimed.descriptor is not None
    assert claimed.descriptor.claim_ownership is not None
    wrong = fixture.store_b.mark_consumed(
        suspended_operation_id=fixture.descriptor.suspended_operation_id,
        expected_materialization_revision=claimed.descriptor.materialization_revision,
        owner_id=OWNER_HOST_B,
        fence=claimed.descriptor.claim_ownership.fence,
    )
    assert wrong.outcome is SuspendedOperationMutationOutcome.STALE_CLAIM


def test_r5_7_sequential_authority_regression_unchanged(tmp_path: Path) -> None:
    from tests.unit.runtime.execution.test_uca6c_r6_r5_7_r2_sequential_authority_closure import (
        test_stale_human_request_id_blocks_resume,
    )

    test_stale_human_request_id_blocks_resume(tmp_path)
