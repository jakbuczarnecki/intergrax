# © Artur Czarnecki. All rights reserved.

"""P0C-7 background worker re-entry convergence proofs."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptLifecycleError, AttemptTransitionReason
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.runtime.background_execution.reentry_admission import (
    BackgroundExecutionReentryAdmissionError,
    BackgroundExecutionReentryDisposition,
    admit_background_execution_reentry,
)
from intergrax.runtime.background_execution.transport_ref import BackgroundTransportExecutionRef
from intergrax.runtime.execution.attempt_lifecycle import AttemptLifecycleService
from intergrax.runtime.execution.execution_terminal import ExecutionTerminalService
from tests.unit.runtime.background_execution.reentry_admission_doubles import (
    InMemoryKVStore,
    make_inmemory_admission_dependencies,
    make_kv_admission_dependencies,
)


pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _transport(*, tenant_id: str = "tenant-a", task_id: str = "transport-1") -> BackgroundTransportExecutionRef:
    return BackgroundTransportExecutionRef(
        tenant_id=tenant_id,
        provider="broker",
        transport_task_id=task_id,
    )


def test_first_admission_establishes_a1_idempotently() -> None:
    deps = make_kv_admission_dependencies()
    transport = _transport()
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    second = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    assert first.disposition is BackgroundExecutionReentryDisposition.EXECUTE
    assert second.disposition is BackgroundExecutionReentryDisposition.EXECUTE
    assert second.identity.task_id == first.identity.task_id
    assert second.identity.run_id == first.identity.run_id
    assert second.identity.attempt_id == first.identity.attempt_id


def test_retry_transition_reentry_uses_active_a2_not_bootstrap_a1() -> None:
    deps = make_kv_admission_dependencies()
    transport = _transport(task_id="retry-recovery")
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    bootstrap_a1 = first.identity.attempt_id
    transition = deps.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=first.identity.tenant_id,
        run_id=first.identity.run_id,
        expected_attempt_id=bootstrap_a1,
        reason=AttemptTransitionReason.RETRY,
    )
    fresh_deps = make_kv_admission_dependencies(kv_store=InMemoryKVStore())
    # Share durable state across fresh service instances (process restart simulation).
    shared_kv = deps.identity_persistence._kv_store  # type: ignore[attr-defined]
    fresh_deps = make_kv_admission_dependencies(kv_store=shared_kv)
    recovered = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=fresh_deps.identity_persistence,
        attempt_lifecycle=fresh_deps.attempt_lifecycle,
        execution_terminal=fresh_deps.execution_terminal,
    )
    assert recovered.identity.attempt_id == transition.active_attempt_id
    assert recovered.identity.attempt_id != bootstrap_a1
    assert recovered.disposition is BackgroundExecutionReentryDisposition.EXECUTE


def test_redelivery_after_a2_does_not_mint_a3() -> None:
    deps = make_kv_admission_dependencies()
    transport = _transport(task_id="no-a3")
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    a2 = deps.attempt_lifecycle.transition_to_next_attempt(
        tenant_id=first.identity.tenant_id,
        run_id=first.identity.run_id,
        expected_attempt_id=first.identity.attempt_id,
        reason=AttemptTransitionReason.RETRY,
    ).active_attempt_id
    attempts = [
        admit_background_execution_reentry(
            transport_ref=transport,
            identity_persistence=deps.identity_persistence,
            attempt_lifecycle=deps.attempt_lifecycle,
            execution_terminal=deps.execution_terminal,
        ).identity.attempt_id
        for _ in range(3)
    ]
    assert attempts == [a2, a2, a2]


@pytest.mark.parametrize(
    "outcome",
    [
        ExecutionTerminalOutcome.COMPLETED,
        ExecutionTerminalOutcome.FAILED,
        ExecutionTerminalOutcome.CANCELLED,
    ],
)
def test_terminal_redelivery_denies_handler_execution(outcome: ExecutionTerminalOutcome) -> None:
    deps = make_inmemory_admission_dependencies()
    transport = _transport(task_id=f"terminal-{outcome.value}")
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    deps.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=outcome,
    )
    redelivery = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    assert redelivery.disposition is BackgroundExecutionReentryDisposition.TERMINAL_ALREADY_RECORDED


def test_corrupt_lifecycle_fails_closed() -> None:
    deps = make_inmemory_admission_dependencies()
    transport = _transport(task_id="corrupt")
    bootstrap = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    deps.attempt_lifecycle.store._records[  # type: ignore[attr-defined]
        (bootstrap.identity.tenant_id, str(bootstrap.identity.run_id))
    ] = b"not-json"
    with pytest.raises(BackgroundExecutionReentryAdmissionError):
        admit_background_execution_reentry(
            transport_ref=transport,
            identity_persistence=deps.identity_persistence,
            attempt_lifecycle=deps.attempt_lifecycle,
            execution_terminal=deps.execution_terminal,
        )


def test_terminal_run_mismatch_fails_closed() -> None:
    deps = make_inmemory_admission_dependencies()
    transport = _transport(task_id="run-mismatch")
    first = admit_background_execution_reentry(
        transport_ref=transport,
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    deps.execution_terminal.commit_terminal_outcome(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=mint_run_id(),
        outcome=ExecutionTerminalOutcome.COMPLETED,
    )
    with pytest.raises(BackgroundExecutionReentryAdmissionError):
        admit_background_execution_reentry(
            transport_ref=transport,
            identity_persistence=deps.identity_persistence,
            attempt_lifecycle=deps.attempt_lifecycle,
            execution_terminal=deps.execution_terminal,
        )


def test_tenant_isolation_for_same_transport_id() -> None:
    kv = InMemoryKVStore()
    deps_a = make_kv_admission_dependencies(kv)
    deps_b = make_kv_admission_dependencies(kv)
    transport_a = _transport(tenant_id="tenant-a", task_id="shared-transport")
    transport_b = _transport(tenant_id="tenant-b", task_id="shared-transport")
    identity_a = admit_background_execution_reentry(
        transport_ref=transport_a,
        identity_persistence=deps_a.identity_persistence,
        attempt_lifecycle=deps_a.attempt_lifecycle,
        execution_terminal=deps_a.execution_terminal,
    ).identity
    identity_b = admit_background_execution_reentry(
        transport_ref=transport_b,
        identity_persistence=deps_b.identity_persistence,
        attempt_lifecycle=deps_b.attempt_lifecycle,
        execution_terminal=deps_b.execution_terminal,
    ).identity
    assert identity_a.task_id != identity_b.task_id
    assert identity_a.run_id != identity_b.run_id


def test_concurrent_first_admission_records_single_a1() -> None:
    deps = make_kv_admission_dependencies()
    transport = _transport(task_id="concurrent-first")
    barrier = threading.Barrier(2)
    results: list[str] = []

    def worker() -> None:
        barrier.wait()
        reentry = admit_background_execution_reentry(
            transport_ref=transport,
            identity_persistence=deps.identity_persistence,
            attempt_lifecycle=deps.attempt_lifecycle,
            execution_terminal=deps.execution_terminal,
        )
        results.append(str(reentry.identity.attempt_id))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(set(results)) == 1


def test_broker_worker_admission_before_started_and_terminal_skip() -> None:
    import base64
    import json

    from intergrax.background_tasks.events import TaskEventName
    from intergrax.contracts.execution_terminal import ExecutionTerminalRecord
    from intergrax.queueing.providers.broker_worker_base import BrokerWorkerBase
    from intergrax.queueing.worker.registry import TaskExecutionRegistry
    from intergrax.runtime.observability.memory_causal_evidence_persistence import (
        InMemoryCausalEvidencePersistence,
    )
    from intergrax.tools.execution_models import ToolExecutionResult
    from pydantic import BaseModel

    class _Output(BaseModel):
        value: str = "ok"

    class _Worker(BrokerWorkerBase):
        def start(self) -> None:
            raise NotImplementedError

    kv = InMemoryKVStore()
    deps = make_kv_admission_dependencies(kv)
    registry = TaskExecutionRegistry()
    handler_called: list[bool] = []

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity,
    ) -> ToolExecutionResult[_Output]:
        handler_called.append(True)
        return ToolExecutionResult.ok(_Output())

    registry.register("demo.task.v1", handler)
    worker = _Worker(
        registry=registry,
        kv_store=kv,
        identity_persistence=deps.identity_persistence,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    message = json.dumps(
        {
            "task_id": "broker-order-1",
            "tenant_id": "tenant-a",
            "run_id": "queue-correlation",
            "task_name": "demo.task.v1",
            "payload": base64.b64encode(b"input").decode("ascii"),
            "idempotency_key": None,
        }
    ).encode("utf-8")

    with patch.object(worker, "_emit_event", wraps=worker._emit_event) as emit_mock:
        worker.process_message(raw_payload=message)
    started_before_handler = any(
        call.args[0] is TaskEventName.STARTED for call in emit_mock.call_args_list
    )
    assert started_before_handler
    assert handler_called == [True]

    first = admit_background_execution_reentry(
        transport_ref=_transport(task_id="broker-order-1"),
        identity_persistence=deps.identity_persistence,
        attempt_lifecycle=deps.attempt_lifecycle,
        execution_terminal=deps.execution_terminal,
    )
    deps.execution_terminal.store._records[  # type: ignore[attr-defined]
        (first.identity.tenant_id, str(first.identity.task_id))
    ] = ExecutionTerminalRecord(
        tenant_id=first.identity.tenant_id,
        task_id=str(first.identity.task_id),
        run_id=first.identity.run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        recorded_at_utc="2026-01-01T00:00:00Z",
    )
    handler_called.clear()
    emit_mock.reset_mock()
    worker.process_message(raw_payload=message)
    assert handler_called == []
    assert TaskEventName.STARTED not in [call.args[0] for call in emit_mock.call_args_list]
