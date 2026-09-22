# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationClaimOutcome,
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
    SuspendedOperationKind,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.contracts.execution.suspended_operation.entity_id import (
    mint_suspended_operation_id,
)
from intergrax.runtime.execution.suspended_operation.in_memory_store import (
    InMemorySuspendedExecutionOperationStore,
)

pytestmark = pytest.mark.unit


def _identity() -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=TaskId("task_" + "a" * 32),
        run_id=RunId("run_" + "b" * 32),
        attempt_id=AttemptId("attempt_" + "c" * 32),
        execution_id=ExecutionId("exec_" + "d" * 32),
    )


def _descriptor() -> SuspendedExecutionOperationDescriptor:
    envelope = SerializedSuspendedOperationEnvelope(
        operation_kind=SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
        payload_schema_version="execution_bound_catalog_tool_payload.v1",
        canonical_json="{}",
    )
    return SuspendedExecutionOperationDescriptor(
        suspended_operation_id=mint_suspended_operation_id(),
        operation_kind=SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
        identity=_identity(),
        continuation_id="gcr_test123",
        invocation_scope_id="dhr_testscope",
        materialization_state=SuspendedOperationMaterializationState.PREPARED,
        materialization_revision=0,
        payload_digest="sha256:" + ("0" * 64),
        payload=envelope,
    )


def test_claim_concurrency_only_one_owner() -> None:
    store = InMemorySuspendedExecutionOperationStore()
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
    lease = datetime.now(UTC) + timedelta(minutes=5)
    first = store.claim(
        suspended_operation_id=descriptor.suspended_operation_id,
        expected_materialization_revision=blocked.descriptor.materialization_revision,
        owner_id="host-a",
        lease_expires_at=lease,
    )
    assert first.outcome is SuspendedOperationClaimOutcome.CLAIMED
    second = store.claim(
        suspended_operation_id=descriptor.suspended_operation_id,
        expected_materialization_revision=blocked.descriptor.materialization_revision,
        owner_id="host-b",
        lease_expires_at=lease,
    )
    assert second.outcome is SuspendedOperationClaimOutcome.STALE_REVISION
