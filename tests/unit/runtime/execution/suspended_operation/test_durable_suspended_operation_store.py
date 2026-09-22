# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationClaimOutcome,
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.entity_id import (
    mint_suspended_operation_id,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.suspended_operation.document_store_suspended_operation_store import (
    DocumentStoreSuspendedExecutionOperationStore,
    reconnect_document_store_suspended_operation_store,
)
from tests.unit.runtime.execution.suspended_operation.test_suspended_operation_store import (
    _descriptor,
    _identity,
)

pytestmark = pytest.mark.unit


def _block_descriptor(store, descriptor):
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
    store.prepare(descriptor)
    return store.block(
        suspended_operation_id=descriptor.suspended_operation_id,
        expected_materialization_revision=0,
        continuation=pending,
        governed_correlation=pending.governed_correlation,
    )


def test_durable_store_survives_process_reconnect() -> None:
    document_store = InMemoryDocumentStore()
    store_a = DocumentStoreSuspendedExecutionOperationStore(document_store)
    descriptor = _descriptor()
    blocked = _block_descriptor(store_a, descriptor)
    assert blocked.outcome is SuspendedOperationMutationOutcome.APPLIED

    store_b = reconnect_document_store_suspended_operation_store(document_store)
    loaded = store_b.load(descriptor.suspended_operation_id)
    assert loaded is not None
    assert loaded.materialization_state.value == "blocked"


def test_reclaim_increments_fence_monotonically() -> None:
    store = DocumentStoreSuspendedExecutionOperationStore(InMemoryDocumentStore())
    descriptor = _descriptor()
    blocked = _block_descriptor(store, descriptor)
    assert blocked.descriptor is not None
    lease = datetime.now(UTC) + timedelta(minutes=5)
    claimed = store.claim(
        suspended_operation_id=descriptor.suspended_operation_id,
        expected_materialization_revision=blocked.descriptor.materialization_revision,
        owner_id="host-a",
        lease_expires_at=lease,
    )
    assert claimed.outcome is SuspendedOperationClaimOutcome.CLAIMED
    assert claimed.descriptor is not None
    assert claimed.descriptor.claim_ownership is not None
    old_fence = claimed.descriptor.claim_ownership.fence
    from intergrax.runtime.execution.suspended_operation import store_engine

    future_now = datetime.now(UTC) + timedelta(hours=1)
    with patch.object(store_engine, "_utc_now", return_value=future_now):
        reclaimed = store.reclaim(
            suspended_operation_id=descriptor.suspended_operation_id,
            expected_materialization_revision=claimed.descriptor.materialization_revision,
            owner_id="host-b",
            lease_expires_at=future_now + timedelta(minutes=5),
            expected_fence=old_fence,
        )
    assert reclaimed.outcome is SuspendedOperationMutationOutcome.APPLIED
    assert reclaimed.descriptor is not None
    assert reclaimed.descriptor.claim_ownership is not None
    assert reclaimed.descriptor.claim_ownership.fence > old_fence
