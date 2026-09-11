# © Artur Czarnecki. All rights reserved.

"""W4-C — distributed external operation cancellation qualification matrix."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future as ConcurrentFuture

import pytest

from intergrax.contracts.dependency_concurrency_admission import (
    DependencyConcurrencyAdmissionRequest,
    DependencyConcurrencyExceededError,
    DependencyConcurrencyIdentity,
    DependencyConcurrencyKind,
    DependencyConcurrencyOverloadMode,
    DependencyConcurrencyPolicy,
)
from intergrax.runtime.resilience.local_dependency_concurrency_admission import (
    LocalDependencyConcurrencyAdmission,
)

from intergrax.contracts.execution_identity import (
    ExecutionId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
)
from intergrax.contracts.external_operation_identity import (
    ExternalOperationIdentity,
    mint_stable_operation_id,
)
from intergrax.runtime.external_operations.external_operation_ownership import (
    ExternalOperationStartSuppressedError,
    ProcessLocalExternalOperationOwner,
    assert_may_begin_physical_attempt,
    mark_physical_running,
    mark_physical_terminal,
    reconcile_orphaned_running_operations,
    request_operation_cancellation,
)
from intergrax.runtime.resilience.dependency_attempt_execution_boundary import (
    DependencyAttemptExecutionBoundary,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateUpdate,
    InMemoryExternalOperationStateStore,
    StaleExternalOperationStateError,
)
from intergrax.runtime.external_operations.recovery_external_operation_gate import (
    ExternalOperationRecoveryBlockedError,
    prepare_external_operations_for_recovery,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _identity(
    *,
    physical_attempt_sequence: int = 1,
    operation_id: str | None = None,
) -> ExternalOperationIdentity:
    execution_id = mint_execution_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        op_id = operation_id or mint_stable_operation_id(
            execution_id=execution_id,
            dependency_kind=DependencyConcurrencyKind.TOOL,
            dependency_identity="demo.tool",
            logical_scope="step-1",
        )
        return ExternalOperationIdentity(
            execution_id=execution_id,
            attempt_id=attempt_id,
            dependency_kind=DependencyConcurrencyKind.TOOL,
            dependency_identity="demo.tool",
            operation_id=op_id,
            physical_attempt_sequence=physical_attempt_sequence,
        )
    finally:
        from intergrax.contracts.execution_identity import reset_active_execution_identity

        reset_active_execution_identity(token)


def test_case_1_cancel_before_start_suppresses_external_call() -> None:
    store = InMemoryExternalOperationStateStore()
    identity = _identity()
    request_operation_cancellation(store, operation_id=identity.operation_id)
    with pytest.raises(ExternalOperationStartSuppressedError):
        assert_may_begin_physical_attempt(store, identity)


def test_case_2_cancel_during_execution_preserves_terminal_state() -> None:
    store = InMemoryExternalOperationStateStore()
    owner_a = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    record = store.create_if_absent(identity.operation_id, owner_token=owner_a.token)
    mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=owner_a,
    )
    cancelled_intent = request_operation_cancellation(
        store,
        operation_id=identity.operation_id,
    )
    terminal = mark_physical_terminal(
        store,
        operation_id=identity.operation_id,
        expected_revision=cancelled_intent.revision,
        physical_state=ExternalOperationPhysicalState.SUCCEEDED,
    )
    loaded = store.load(identity.operation_id)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.SUCCEEDED
    assert terminal.revision == loaded.revision
    with pytest.raises(StaleExternalOperationStateError):
        mark_physical_terminal(
            store,
            operation_id=identity.operation_id,
            expected_revision=cancelled_intent.revision,
            physical_state=ExternalOperationPhysicalState.FAILED,
        )


def test_case_3_worker_crash_marks_unknown() -> None:
    store = InMemoryExternalOperationStateStore()
    dead_owner = ProcessLocalExternalOperationOwner.mint()
    active_owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    record = store.create_if_absent(identity.operation_id, owner_token=dead_owner.token)
    running = mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=dead_owner,
    )
    assert running.physical_state is ExternalOperationPhysicalState.RUNNING
    reconciled = reconcile_orphaned_running_operations(
        store,
        active_owner=active_owner,
    )
    assert len(reconciled) == 1
    assert reconciled[0].physical_state is ExternalOperationPhysicalState.UNKNOWN


def test_case_4_double_cancellation_is_idempotent() -> None:
    store = InMemoryExternalOperationStateStore()
    identity = _identity()
    first = request_operation_cancellation(store, operation_id=identity.operation_id)
    second = request_operation_cancellation(store, operation_id=identity.operation_id)
    assert first.revision == second.revision
    assert second.intent_state is ExternalOperationIntentState.CANCELLATION_REQUESTED


def test_case_5_retry_after_cancellation_blocked() -> None:
    store = InMemoryExternalOperationStateStore()
    identity = _identity(physical_attempt_sequence=2)
    request_operation_cancellation(store, operation_id=identity.operation_id)
    with pytest.raises(ExternalOperationStartSuppressedError):
        assert_may_begin_physical_attempt(store, identity)


def test_recovery_blocked_on_unknown_until_reconciled() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    record = store.create_if_absent(identity.operation_id, owner_token="stale")
    mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=ProcessLocalExternalOperationOwner(token="stale"),
    )
    with pytest.raises(ExternalOperationRecoveryBlockedError):
        prepare_external_operations_for_recovery(
            store,
            active_owner=owner,
            operation_ids=(identity.operation_id,),
        )
    prepare_external_operations_for_recovery(store, active_owner=owner)
    loaded = store.load(identity.operation_id)
    assert loaded is not None
    assert loaded.physical_state is ExternalOperationPhysicalState.UNKNOWN


class _RecordingCancelPort:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def request_cancel(self, operation_id: str) -> None:
        self.calls.append(operation_id)


def test_cancellation_port_invoked_once() -> None:
    store = InMemoryExternalOperationStateStore()
    port = _RecordingCancelPort()
    identity = _identity()
    request_operation_cancellation(
        store,
        operation_id=identity.operation_id,
        cancellation_port=port,
    )
    assert port.calls == [identity.operation_id]


def test_stable_operation_id_across_physical_attempts() -> None:
    execution_id = ExecutionId("exec_" + "a" * 32)
    first = mint_stable_operation_id(
        execution_id=execution_id,
        dependency_kind=DependencyConcurrencyKind.TOOL,
        dependency_identity="pay",
        logical_scope="payment-123",
    )
    second = mint_stable_operation_id(
        execution_id=execution_id,
        dependency_kind=DependencyConcurrencyKind.TOOL,
        dependency_identity="pay",
        logical_scope="payment-123",
    )
    assert first == second


def test_identity_retry_same_operation_id_different_physical_sequence() -> None:
    execution_id = ExecutionId("exec_" + "b" * 32)
    op = mint_stable_operation_id(
        execution_id=execution_id,
        dependency_kind=DependencyConcurrencyKind.TOOL,
        dependency_identity="payment",
        logical_scope="payment-request-123",
    )
    id1 = ExternalOperationIdentity(
        execution_id=execution_id,
        attempt_id=mint_attempt_id(),
        dependency_kind=DependencyConcurrencyKind.TOOL,
        dependency_identity="payment",
        operation_id=op,
        physical_attempt_sequence=1,
    )
    id2 = ExternalOperationIdentity(
        execution_id=execution_id,
        attempt_id=mint_attempt_id(),
        dependency_kind=DependencyConcurrencyKind.TOOL,
        dependency_identity="payment",
        operation_id=op,
        physical_attempt_sequence=2,
    )
    assert id1.operation_id == id2.operation_id
    assert id1.attempt_id != id2.attempt_id
    assert id1.physical_attempt_sequence != id2.physical_attempt_sequence


def test_triple_cancel_idempotent_single_logical_transition() -> None:
    store = InMemoryExternalOperationStateStore()
    identity = _identity()
    revisions: list[int] = []
    for _ in range(3):
        record = request_operation_cancellation(store, operation_id=identity.operation_id)
        revisions.append(record.revision)
    assert revisions[0] == revisions[1] == revisions[2]
    loaded = store.load(identity.operation_id)
    assert loaded is not None
    assert loaded.intent_state is ExternalOperationIntentState.CANCELLATION_REQUESTED


def test_triple_cancel_does_not_invoke_cancellation_port_repeatedly() -> None:
    store = InMemoryExternalOperationStateStore()
    port = _RecordingCancelPort()
    identity = _identity()
    for _ in range(3):
        request_operation_cancellation(
            store,
            operation_id=identity.operation_id,
            cancellation_port=port,
        )
    assert port.calls == [identity.operation_id]


def test_retry_allowed_after_physical_failed() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    first = _identity(physical_attempt_sequence=1)
    identity = _identity(
        operation_id=first.operation_id,
        physical_attempt_sequence=2,
    )
    record = store.create_if_absent(identity.operation_id, owner_token=owner.token)
    running = mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=owner,
    )
    failed = mark_physical_terminal(
        store,
        operation_id=identity.operation_id,
        expected_revision=running.revision,
        physical_state=ExternalOperationPhysicalState.FAILED,
    )
    assert failed.physical_state is ExternalOperationPhysicalState.FAILED
    assert_may_begin_physical_attempt(store, identity)


def test_cas_terminal_race_twenty_writers_one_winner() -> None:
    store = InMemoryExternalOperationStateStore()
    owner = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    record = store.create_if_absent(identity.operation_id, owner_token=owner.token)
    running = mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=owner,
    )
    barrier = threading.Barrier(20)
    cas_success_count = 0
    stale_count = 0
    lock = threading.Lock()

    def writer(physical: ExternalOperationPhysicalState) -> None:
        nonlocal stale_count, cas_success_count
        barrier.wait()
        try:
            store.compare_and_set(
                identity.operation_id,
                expected_revision=running.revision,
                update=ExternalOperationStateUpdate(physical_state=physical),
            )
        except StaleExternalOperationStateError:
            with lock:
                stale_count += 1
            return
        with lock:
            cas_success_count += 1

    threads = [
        threading.Thread(
            target=writer,
            args=(
                ExternalOperationPhysicalState.SUCCEEDED
                if i % 2 == 0
                else ExternalOperationPhysicalState.CANCELLED,
            ),
        )
        for i in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert cas_success_count == 1
    assert stale_count == 19
    final = store.load(identity.operation_id)
    assert final is not None
    assert final.physical_state in (
        ExternalOperationPhysicalState.SUCCEEDED,
        ExternalOperationPhysicalState.CANCELLED,
    )
    assert final.revision == running.revision + 1


def _tool_admission(capacity: int) -> LocalDependencyConcurrencyAdmission:
    identity = DependencyConcurrencyIdentity(
        kind=DependencyConcurrencyKind.TOOL,
        value="w4c.permit.tool",
    )
    policy = DependencyConcurrencyPolicy(
        max_concurrent_calls=capacity,
        overload_mode=DependencyConcurrencyOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )
    return LocalDependencyConcurrencyAdmission({identity: policy})


def _tool_request() -> DependencyConcurrencyAdmissionRequest:
    return DependencyConcurrencyAdmissionRequest(
        dependency=DependencyConcurrencyIdentity(
            kind=DependencyConcurrencyKind.TOOL,
            value="w4c.permit.tool",
        ),
        tenant_id=None,
    )


@pytest.mark.asyncio
async def test_cancel_during_worker_execution_blocks_second_acquire_until_terminal() -> None:
    admission = _tool_admission(1)
    store = InMemoryExternalOperationStateStore()
    owner_a = ProcessLocalExternalOperationOwner.mint()
    identity = _identity()
    record = store.create_if_absent(identity.operation_id, owner_token=owner_a.token)
    permit_a = await admission.acquire(_tool_request())
    mark_physical_running(
        store,
        operation_id=identity.operation_id,
        expected_revision=record.revision,
        owner=owner_a,
    )
    after_cancel = request_operation_cancellation(
        store,
        operation_id=identity.operation_id,
    )
    with pytest.raises(DependencyConcurrencyExceededError):
        await admission.acquire(_tool_request())
    mark_physical_terminal(
        store,
        operation_id=identity.operation_id,
        expected_revision=after_cancel.revision,
        physical_state=ExternalOperationPhysicalState.CANCELLED,
    )
    await permit_a.release()
    permit_b = await admission.acquire(_tool_request())
    await permit_b.release()


@pytest.mark.asyncio
async def test_permit_no_leak_after_cancel_exception_timeout_shutdown_paths() -> None:
    admission = _tool_admission(1)
    boundary = DependencyAttemptExecutionBoundary(admission)
    request = _tool_request()

    handle = boundary.acquire(request)
    boundary.release_after_submit_failure(handle)

    handle = boundary.acquire(request)
    worker = ConcurrentFuture()
    boundary.bind_worker(handle, worker)
    worker.set_exception(RuntimeError("tool failed"))
    boundary.complete_attached(handle)

    handle = boundary.acquire(request)
    worker = ConcurrentFuture()
    boundary.bind_worker(handle, worker)
    assert boundary.detach_if_still_running(handle)
    worker.set_result(None)
    await asyncio.sleep(0.05)

    handle = boundary.acquire(request)
    worker = ConcurrentFuture()
    boundary.bind_worker(handle, worker)
    worker.set_result(None)
    boundary.complete_attached(handle)

    permit = await admission.acquire(request)
    await permit.release()
    boundary.close()
