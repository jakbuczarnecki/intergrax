# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
    RuntimeHistoryCompleteness,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_PROVIDER = "celery"


def _reconstructor(
    runtime_store: RuntimeEventPersistence | None = None,
    causal_store: CausalEvidencePersistence | None = None,
) -> ExecutionReconstructor:
    return ExecutionReconstructor(
        runtime_events=runtime_store or InMemoryRuntimeEventStore(),
        causal_evidence=causal_store or InMemoryCausalEvidencePersistence(),
    )


def _transport_ref(*, tenant_id: str = _TENANT_A, transport_task_id: str = "celery-task-1") -> MessageBusTaskRef:
    return MessageBusTaskRef(
        provider=_PROVIDER,
        task_id=transport_task_id,
        tenant_id=tenant_id,
    )


def _execution_ref(
    *,
    tenant_id: str = _TENANT_A,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
) -> RuntimeExecutionRef:
    return RuntimeExecutionRef(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id=tenant_id,
    )


def _causal_evidence(
    *,
    tenant_id: str = _TENANT_A,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    transport_task_id: str = "celery-task-1",
    recorded_at: datetime | None = None,
) -> PlatformCausalEvidence:
    return PlatformCausalEvidence(
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=tenant_id,
        source=_transport_ref(tenant_id=tenant_id, transport_task_id=transport_task_id),
        target=_execution_ref(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        ),
        recorded_at=recorded_at or datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
    )


def _append_event(
    store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    timestamp: datetime | None = None,
) -> None:
    event = sample_runtime_event(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    if timestamp is not None:
        event = event.model_copy(update={"timestamp": timestamp})
    store.append(event, tenant_id=tenant_id)


def test_basic_reconstruction_one_attempt_multiple_events() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    evidence = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    causal_store.append(evidence)
    for offset in (0, 1, 2):
        _append_event(
            runtime_store,
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            timestamp=datetime(2026, 6, 8, 12, offset, 0, tzinfo=timezone.utc),
        )

    reconstruction = _reconstructor(runtime_store, causal_store).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )

    assert reconstruction.tenant_id == _TENANT_A
    assert reconstruction.task_id == task_id
    assert reconstruction.run_id == run_id
    assert reconstruction.causal_evidence == (evidence,)
    assert len(reconstruction.positioned_events) == 3
    assert [row.position.value for row in reconstruction.positioned_events] == [1, 2, 3]
    assert len(reconstruction.attempts) == 1
    attempt = reconstruction.attempts[0]
    assert attempt.attempt_id == attempt_id
    assert attempt.causal_evidence == (evidence,)
    assert attempt.positioned_events == reconstruction.positioned_events
    assert reconstruction.is_runtime_history_complete


def test_retry_two_attempts_separated() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    evidence_a1 = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        transport_task_id="transport-1",
        recorded_at=datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
    )
    evidence_a2 = _causal_evidence(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a2,
        transport_task_id="transport-2",
        recorded_at=datetime(2026, 6, 8, 12, 5, 0, tzinfo=timezone.utc),
    )
    causal_store.append(evidence_a1)
    causal_store.append(evidence_a2)
    _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_a1)
    _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_a2)
    _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_a2)

    reconstruction = _reconstructor(runtime_store, causal_store).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )

    assert len(reconstruction.attempts) == 2
    assert reconstruction.attempts[0].attempt_id == attempt_a1
    assert reconstruction.attempts[1].attempt_id == attempt_a2
    assert reconstruction.attempts[0].causal_evidence == (evidence_a1,)
    assert reconstruction.attempts[1].causal_evidence == (evidence_a2,)
    assert len(reconstruction.attempts[0].positioned_events) == 1
    assert len(reconstruction.attempts[1].positioned_events) == 2
    assert [row.position.value for row in reconstruction.positioned_events] == [1, 2, 3]


def test_evidence_only_attempt_retained() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    causal_store = InMemoryCausalEvidencePersistence()
    evidence = _causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_a1)
    causal_store.append(evidence)

    reconstruction = _reconstructor(causal_store=causal_store).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )

    assert len(reconstruction.attempts) == 1
    assert reconstruction.attempts[0].attempt_id == attempt_a1
    assert reconstruction.attempts[0].causal_evidence == (evidence,)
    assert reconstruction.attempts[0].positioned_events == ()
    assert reconstruction.positioned_events == ()


def test_event_only_attempt_retained() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a2 = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_a2)

    reconstruction = _reconstructor(runtime_store).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )

    assert len(reconstruction.attempts) == 1
    assert reconstruction.attempts[0].attempt_id == attempt_a2
    assert reconstruction.attempts[0].causal_evidence == ()
    assert len(reconstruction.attempts[0].positioned_events) == 1


def test_attempt_union_causal_a1_a2_runtime_a2_a3() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    attempt_a3 = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_a1))
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_a2))
    _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_a2)
    _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_a3)

    reconstruction = _reconstructor(runtime_store, causal_store).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )

    assert {attempt.attempt_id for attempt in reconstruction.attempts} == {
        attempt_a1,
        attempt_a2,
        attempt_a3,
    }


def test_tenant_isolation() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(
        _causal_evidence(tenant_id=_TENANT_B, task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    )
    _append_event(
        runtime_store,
        tenant_id=_TENANT_B,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )

    reconstruction = _reconstructor(runtime_store, causal_store).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )

    assert reconstruction.causal_evidence == ()
    assert reconstruction.positioned_events == ()
    assert reconstruction.attempts == ()


class _CorruptCausalPersistence(InMemoryCausalEvidencePersistence):
    def list_for_execution(self, *, tenant_id: str, task_id: TaskId, run_id: RunId):
        records = super().list_for_execution(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
        )
        if not records:
            return records
        first = records[0]
        corrupted = first.model_copy(
            update={
                "target": first.target.model_copy(
                    update={"run_id": mint_run_id()},
                ),
            },
        )
        return (corrupted,)


class _CorruptRuntimePersistence(InMemoryRuntimeEventStore):
    def list_positioned_for_run(self, run_id: str, *, tenant_id: str, limit: int = 1000, through=None):
        rows = super().list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=through,
        )
        if not rows:
            return rows
        first = rows[0]
        corrupted_event = first.event.model_copy(update={"run_id": mint_run_id()})
        return [
            type(first)(event=corrupted_event, position=first.position),
            *rows[1:],
        ]


def test_corrupted_causal_persistence_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    causal_store = _CorruptCausalPersistence()
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id))

    with pytest.raises(ExecutionReconstructionIntegrityError, match="run_id"):
        _reconstructor(causal_store=causal_store).reconstruct_execution(_TENANT_A, task_id, run_id)


def test_corrupted_runtime_persistence_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = _CorruptRuntimePersistence()
    _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_id)

    with pytest.raises(ExecutionReconstructionIntegrityError, match="run_id"):
        _reconstructor(runtime_store).reconstruct_execution(_TENANT_A, task_id, run_id)


def test_ordering_follows_execution_position_not_timestamp() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=2), base, base + timedelta(hours=1)]
    for timestamp in timestamps:
        _append_event(
            runtime_store,
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            timestamp=timestamp,
        )

    reconstruction = _reconstructor(runtime_store).reconstruct_execution(_TENANT_A, task_id, run_id)

    assert [row.position.value for row in reconstruction.positioned_events] == [1, 2, 3]
    assert [row.event.timestamp for row in reconstruction.positioned_events] == timestamps


def test_determinism_same_state_same_output() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(_causal_evidence(task_id=task_id, run_id=run_id, attempt_id=attempt_id))
    _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_id)
    reconstructor = _reconstructor(runtime_store, causal_store)

    first = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    second = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)

    assert first == second


def test_runtime_history_truncation_exposed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    for _ in range(5):
        _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_id)

    reconstruction = _reconstructor(runtime_store).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
        initial_limit=2,
        max_limit=2,
    )

    assert reconstruction.runtime_history_completeness is RuntimeHistoryCompleteness.TRUNCATED
    assert len(reconstruction.positioned_events) == 2
    assert reconstruction.is_runtime_history_complete is False


def test_runtime_history_complete_when_within_limits() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    for _ in range(3):
        _append_event(runtime_store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, attempt_id=attempt_id)

    reconstruction = _reconstructor(runtime_store).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
        initial_limit=2,
        max_limit=10,
    )

    assert reconstruction.runtime_history_completeness is RuntimeHistoryCompleteness.COMPLETE
    assert len(reconstruction.positioned_events) == 3
