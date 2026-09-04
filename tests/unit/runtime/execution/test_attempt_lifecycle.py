# © Artur Czarnecki. All rights reserved.

"""P0C-4 durable attempt lifecycle proofs."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.attempt_lifecycle import (
    AttemptLifecycleError,
    AttemptTransitionReason,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    peek_active_execution_identity,
    rebind_active_attempt_for_retry,
    reset_active_execution_identity,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
    decode_attempt_lifecycle_state,
    encode_attempt_lifecycle_state,
)


@pytest.mark.unit
def test_retry_transition_mints_new_attempt_same_run() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    service.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)

    result = service.transition_to_next_attempt(
        tenant_id=tenant_id,
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        reason=AttemptTransitionReason.RETRY,
    )

    assert result.previous_attempt_id == attempt_a1
    assert result.active_attempt_id != attempt_a1
    assert result.run_id == run_id
    assert service.get_active_attempt_id(tenant_id=tenant_id, run_id=run_id) == result.active_attempt_id


@pytest.mark.unit
def test_stale_predecessor_rejected_without_new_attempt() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = service.transition_to_next_attempt(
        tenant_id=tenant_id,
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id

    with pytest.raises(StaleClaimError):
        service.transition_to_next_attempt(
            tenant_id=tenant_id,
            run_id=run_id,
            expected_attempt_id=attempt_a1,
            reason=AttemptTransitionReason.RETRY,
        )

    assert service.get_active_attempt_id(tenant_id=tenant_id, run_id=run_id) == attempt_a2


@pytest.mark.unit
def test_concurrent_transition_creates_single_next_attempt() -> None:
    store = InMemoryAttemptLifecycleStore()
    service = AttemptLifecycleService(store)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    barrier = threading.Barrier(2)
    results: list[str] = []
    errors: list[BaseException] = []

    def worker() -> None:
        barrier.wait()
        try:
            result = service.transition_to_next_attempt(
                tenant_id=tenant_id,
                run_id=run_id,
                expected_attempt_id=attempt_a1,
                reason=AttemptTransitionReason.RETRY,
            )
            results.append(str(result.active_attempt_id))
        except BaseException as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], StaleClaimError)
    active = service.get_active_attempt_id(tenant_id=tenant_id, run_id=run_id)
    assert active == results[0]


@pytest.mark.unit
def test_store_failure_does_not_change_active_context() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a1)
    service._store.compare_and_swap = MagicMock(side_effect=RuntimeError("store down"))  # type: ignore[method-assign]
    try:
        with pytest.raises(AttemptLifecycleError):
            service.transition_to_next_attempt(
                tenant_id="tenant-a",
                run_id=run_id,
                expected_attempt_id=attempt_a1,
                reason=AttemptTransitionReason.RETRY,
            )
        assert peek_active_execution_identity() == (run_id, attempt_a1)
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
def test_tenant_isolation_for_same_run_id_string() -> None:
    service = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt_t1 = mint_attempt_id()
    attempt_t2 = mint_attempt_id()
    service.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_t1)
    service.record_initial_attempt(tenant_id="tenant-b", run_id=run_id, attempt_id=attempt_t2)

    assert service.get_active_attempt_id(tenant_id="tenant-a", run_id=run_id) == attempt_t1
    assert service.get_active_attempt_id(tenant_id="tenant-b", run_id=run_id) == attempt_t2


@pytest.mark.unit
def test_corrupt_record_fails_closed() -> None:
    store = InMemoryAttemptLifecycleStore()
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    store._records[(tenant_id, str(run_id))] = b"not-json"  # type: ignore[attr-defined]
    service = AttemptLifecycleService(store)
    with pytest.raises(AttemptLifecycleError):
        service.get_active_attempt_id(tenant_id=tenant_id, run_id=run_id)


@pytest.mark.unit
def test_rebind_active_attempt_for_retry_preserves_run_id() -> None:
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a1)
    attempt_a2 = mint_attempt_id()
    rebound = rebind_active_attempt_for_retry(run_id=run_id, attempt_id=attempt_a2)
    assert rebound == attempt_a2
    assert peek_active_execution_identity() == (run_id, attempt_a2)
    reset_active_execution_identity(token)


@pytest.mark.unit
def test_round_trip_encoding() -> None:
    from intergrax.contracts.attempt_lifecycle import AttemptLifecycleState

    state = AttemptLifecycleState(
        run_id=mint_run_id(),
        active_attempt_id=mint_attempt_id(),
        previous_attempt_id=mint_attempt_id(),
        generation=2,
        transition_reason=AttemptTransitionReason.RETRY,
    )
    decoded = decode_attempt_lifecycle_state(encode_attempt_lifecycle_state(state))
    assert decoded == state
