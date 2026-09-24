# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R5.9-R1-R1 — durable restart identity cross-source correlation."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution.suspended_operation.codec import (
    SerializedSuspendedOperationEnvelope,
    SuspendedOperationKind,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.execution.suspended_operation.entity_id import (
    mint_suspended_operation_id,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.runtime.execution.continuation.durable_restart_identity_correlation import (
    correlate_durable_restart_execution_identity,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.task.task_state import TaskState
from testing_support.builder import (
    canonical_run_id_for_tests,
    canonical_task_id_for_tests,
)

pytestmark = pytest.mark.unit

_TASK = canonical_task_id_for_tests("uca6c-id-corr")
_RUN = canonical_run_id_for_tests("uca6c-id-corr")
_OTHER_RUN = canonical_run_id_for_tests("uca6c-id-corr-other")
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONT = "cont_uca6c_corr"


def _checkpoint() -> TaskCheckpoint:
    tree = ExecutionTreeSnapshot(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        entries=(
            ExecutionCheckpointEntry(
                execution_id=_EXECUTION,
                parent_execution_id=None,
                status=ExecutionCheckpointStatus.RUNNING,
            ),
        ),
    )
    return TaskCheckpoint(
        task_id=str(_TASK),
        tenant_id="tenant-uca6c",
        resume_token="resume-token",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=RuntimeCheckpoint(
            run_id=_RUN,
            attempt_id=_ATTEMPT,
            execution_tree=tree,
        ),
    )


def _descriptor(
    identity: ExecutionContinuationIdentity,
) -> SuspendedExecutionOperationDescriptor:
    envelope = SerializedSuspendedOperationEnvelope(
        operation_kind=SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
        payload_schema_version="execution_bound_catalog_tool_payload.v1",
        canonical_json="{}",
    )
    return SuspendedExecutionOperationDescriptor(
        suspended_operation_id=mint_suspended_operation_id(),
        operation_kind=SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
        identity=identity,
        continuation_id=_CONT,
        invocation_scope_id="dhr_uca6c_corr",
        materialization_state=SuspendedOperationMaterializationState.BLOCKED,
        materialization_revision=0,
        payload_digest="sha256:" + ("0" * 64),
        payload=envelope,
        authority_scope=SuspendedOperationAuthorityScope.DECLARATIVE_GOVERNANCE,
    )


def _store_with_pause() -> InMemoryExecutionContinuationStateStore:
    store = InMemoryExecutionContinuationStateStore()
    identity = ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    pending = PendingExecutionContinuation(
        continuation_id=_CONT,
        identity=identity,
        lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=1,
        reason=ContinuationReason.SECURITY,
        pause_id="pause",
        human_request_id="hr",
    )
    assert store.begin_current_episode_if_predecessor_allows(pending)
    return store


def test_correlate_durable_restart_identity_exact_match() -> None:
    identity = ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    correlated = correlate_durable_restart_execution_identity(
        checkpoint=_checkpoint(),
        continuation_store=_store_with_pause(),
        continuation_id=_CONT,
        suspended_descriptor=_descriptor(identity),
    )
    assert correlated.run_id == _RUN
    assert correlated.execution_id == _EXECUTION


def test_correlate_durable_restart_identity_mismatch_fail_closed() -> None:
    mismatched = ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_OTHER_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    with pytest.raises(ExecutionContinuationError) as exc:
        correlate_durable_restart_execution_identity(
            checkpoint=_checkpoint(),
            continuation_store=_store_with_pause(),
            continuation_id=_CONT,
            suspended_descriptor=_descriptor(mismatched),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.IDENTITY_MISMATCH
