# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Conformance harness for observability persistence backends (OBS-BUS-5, DIAG-1P1)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.execution_position import PositionedRuntimeEvent
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
    CausalEvidencePersistenceConflictError,
    causal_evidence_query_order_key,
)


def sample_runtime_event(
    *,
    event_id: EventId | None = None,
    tenant_id: str = "tenant-conformance",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> RuntimeEvent:
    resolved_task_id = task_id or mint_task_id()
    resolved_run_id = run_id or mint_run_id()
    return RuntimeEvent(
        event_id=event_id or mint_event_id(),
        tenant_id=tenant_id,
        task_id=resolved_task_id,
        run_id=resolved_run_id,
        attempt_id=attempt_id or mint_attempt_id(),
        execution_id=execution_id or mint_execution_id(),
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
        timestamp=datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
        correlation_id=resolved_task_id,
    )


def assert_runtime_event_persistence_conformance(
    store: RuntimeEventPersistence,
    *,
    label: str,
) -> None:
    """
    Shared behavioral contract for every ``RuntimeEventPersistence`` backend.

    Covers tenant scoping, run/task listing, and idempotent append on ``event_id``.
    """
    tenant_a = f"{label}-tenant-a"
    tenant_b = f"{label}-tenant-b"
    run_id = mint_run_id()
    task_id = mint_task_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()

    first = sample_runtime_event(
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    second = sample_runtime_event(
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    foreign = sample_runtime_event(
        tenant_id=tenant_b,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )

    store.append(first, tenant_id=tenant_a)
    store.append(second, tenant_id=tenant_a)
    store.append(foreign, tenant_id=tenant_b)
    duplicate = store.append(first, tenant_id=tenant_a)

    by_run = store.list_for_run(run_id, tenant_id=tenant_a)
    positioned = store.list_positioned_for_run(run_id, tenant_id=tenant_a)
    by_task = store.list_for_task(task_id, tenant_id=tenant_a)
    assert len(by_run) == 2, f"{label}: expected 2 run events, got {len(by_run)}"
    assert len(positioned) == 2, f"{label}: expected 2 positioned run events, got {len(positioned)}"
    assert len(by_task) == 2, f"{label}: expected 2 task events, got {len(by_task)}"
    assert {evt.event_id for evt in by_run} == {first.event_id, second.event_id}
    assert all(evt.tenant_id == tenant_a for evt in by_run)
    assert positioned[0].position.value < positioned[1].position.value
    assert duplicate.position == positioned[0].position
    assert store.list_for_run(run_id, tenant_id=tenant_b) == []
    assert store.list_for_task(task_id, tenant_id=tenant_b) == []

    assert_runtime_event_get_by_event_id_conformance(
        store,
        label=label,
        tenant_a=tenant_a,
        tenant_b=tenant_b,
        first=first,
        duplicate=duplicate,
        positioned_first=positioned[0],
    )


def assert_runtime_event_get_by_event_id_conformance(
    store: RuntimeEventPersistence,
    *,
    label: str,
    tenant_a: str,
    tenant_b: str,
    first: RuntimeEvent,
    duplicate: PositionedRuntimeEvent,
    positioned_first: PositionedRuntimeEvent,
) -> None:
    """Shared ``get_by_event_id`` contract for every ``RuntimeEventPersistence`` backend."""
    lookup = store.get_by_event_id(tenant_id=tenant_a, event_id=first.event_id)
    assert lookup is not None, f"{label}: expected positioned event for accepted event_id"
    assert lookup.event.event_id == first.event_id
    assert lookup.position == positioned_first.position

    unknown_id = mint_event_id()
    assert store.get_by_event_id(tenant_id=tenant_a, event_id=unknown_id) is None

    assert store.get_by_event_id(tenant_id=tenant_b, event_id=first.event_id) is None

    duplicate_lookup = store.get_by_event_id(tenant_id=tenant_a, event_id=first.event_id)
    assert duplicate_lookup is not None
    assert duplicate_lookup.position == duplicate.position

    repeat_lookup = store.get_by_event_id(tenant_id=tenant_a, event_id=first.event_id)
    assert repeat_lookup == duplicate_lookup

    assert lookup.position.value == positioned_first.position.value

    snapshot_event_id = lookup.event.event_id
    snapshot_position = lookup.position.value
    after_lookup = store.get_by_event_id(tenant_id=tenant_a, event_id=first.event_id)
    assert after_lookup is not None
    assert after_lookup.event.event_id == snapshot_event_id
    assert after_lookup.position.value == snapshot_position


def sample_causal_evidence(
    *,
    evidence_id: EventId | None = None,
    tenant_id: str = "tenant-conformance",
    provider: str = "celery",
    transport_task_id: str = "transport-task-conformance",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
) -> PlatformCausalEvidence:
    resolved_task_id = task_id or mint_task_id()
    resolved_run_id = run_id or mint_run_id()
    return PlatformCausalEvidence(
        evidence_id=evidence_id or mint_event_id(),
        relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
        tenant_id=tenant_id,
        source=MessageBusTaskRef(
            provider=provider,
            task_id=transport_task_id,
            tenant_id=tenant_id,
        ),
        target=RuntimeExecutionRef(
            task_id=resolved_task_id,
            run_id=resolved_run_id,
            attempt_id=attempt_id or mint_attempt_id(),
            tenant_id=tenant_id,
        ),
    )


def assert_causal_evidence_persistence_conformance(
    store: CausalEvidencePersistence,
    *,
    label: str,
) -> None:
    """
    Shared behavioral contract for every ``CausalEvidencePersistence`` backend.

    Covers append/read semantics, idempotency, tenant isolation, transport opacity,
    and 1:N relations.
    """
    tenant_a = f"{label}-tenant-a"
    tenant_b = f"{label}-tenant-b"
    provider = "celery"
    transport_task_id = f"{label}-transport-task"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    primary = sample_causal_evidence(
        tenant_id=tenant_a,
        provider=provider,
        transport_task_id=transport_task_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    secondary = sample_causal_evidence(
        tenant_id=tenant_a,
        provider=provider,
        transport_task_id=transport_task_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
    )
    foreign = sample_causal_evidence(
        tenant_id=tenant_b,
        provider=provider,
        transport_task_id=transport_task_id,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )

    stored_primary = store.append(primary)
    store.append(secondary)
    store.append(foreign)
    duplicate = store.append(primary)

    assert stored_primary == primary
    assert duplicate == primary

    by_execution = store.list_for_execution(
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
    )
    by_transport = store.list_for_transport_task(
        tenant_id=tenant_a,
        provider=provider,
        transport_task_id=transport_task_id,
    )
    assert len(by_execution) == 2, f"{label}: expected 2 execution records, got {len(by_execution)}"
    assert len(by_transport) == 2, f"{label}: expected 2 transport records, got {len(by_transport)}"
    assert {record.evidence_id for record in by_execution} == {
        primary.evidence_id,
        secondary.evidence_id,
    }
    assert {record.evidence_id for record in by_transport} == {
        primary.evidence_id,
        secondary.evidence_id,
    }
    assert all(record.tenant_id == tenant_a for record in by_execution)
    assert all(record.tenant_id == tenant_a for record in by_transport)
    expected = tuple(
        sorted(
            (primary, secondary),
            key=causal_evidence_query_order_key,
        )
    )
    assert by_execution == expected
    assert by_transport == expected

    assert store.list_for_execution(
        tenant_id=tenant_b,
        task_id=task_id,
        run_id=run_id,
    ) == ()
    assert store.list_for_transport_task(
        tenant_id=tenant_a,
        provider=provider,
        transport_task_id=transport_task_id,
    ) == by_transport
    assert store.list_for_transport_task(
        tenant_id=tenant_b,
        provider=provider,
        transport_task_id=transport_task_id,
    ) == (foreign,)
    assert store.list_for_execution(
        tenant_id=tenant_b,
        task_id=foreign.target.task_id,
        run_id=foreign.target.run_id,
    ) == (foreign,)

    opaque_transport_id = str(task_id)
    opaque = sample_causal_evidence(
        tenant_id=tenant_a,
        provider=provider,
        transport_task_id=opaque_transport_id,
        task_id=task_id,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )
    store.append(opaque)
    assert store.list_for_transport_task(
        tenant_id=tenant_a,
        provider=provider,
        transport_task_id=opaque_transport_id,
    ) == (opaque,)
    assert store.list_for_execution(
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=opaque.target.run_id,
    ) == (opaque,)

    assert store.list_for_execution(
        tenant_id=tenant_a,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    ) == ()
    assert store.list_for_transport_task(
        tenant_id=tenant_a,
        provider="unknown-provider",
        transport_task_id="missing-transport",
    ) == ()


def assert_causal_evidence_conflicting_append_fails_closed(
    store: CausalEvidencePersistence,
    *,
    label: str,
) -> None:
    tenant_id = f"{label}-conflict-tenant"
    evidence_id = mint_event_id()
    original = sample_causal_evidence(tenant_id=tenant_id, evidence_id=evidence_id)
    conflicting = sample_causal_evidence(
        tenant_id=tenant_id,
        evidence_id=evidence_id,
        transport_task_id="different-transport",
    )

    store.append(original)
    try:
        store.append(conflicting)
    except CausalEvidencePersistenceConflictError:
        pass
    else:
        raise AssertionError(f"{label}: expected conflicting append to fail closed")

    by_execution = store.list_for_execution(
        tenant_id=tenant_id,
        task_id=original.target.task_id,
        run_id=original.target.run_id,
    )
    assert by_execution == (original,)


def assert_causal_evidence_provider_isolation(
    store: CausalEvidencePersistence,
    *,
    label: str,
) -> None:
    tenant_id = f"{label}-provider-tenant"
    transport_task_id = f"{label}-shared-transport-id"
    celery = sample_causal_evidence(
        tenant_id=tenant_id,
        provider="celery",
        transport_task_id=transport_task_id,
    )
    rabbitmq = sample_causal_evidence(
        tenant_id=tenant_id,
        provider="rabbitmq",
        transport_task_id=transport_task_id,
    )
    store.append(celery)
    store.append(rabbitmq)

    assert store.list_for_transport_task(
        tenant_id=tenant_id,
        provider="celery",
        transport_task_id=transport_task_id,
    ) == (celery,)
    assert store.list_for_transport_task(
        tenant_id=tenant_id,
        provider="rabbitmq",
        transport_task_id=transport_task_id,
    ) == (rabbitmq,)


def assert_causal_evidence_typed_round_trip(
    store: CausalEvidencePersistence,
    *,
    label: str,
) -> None:
    tenant_id = f"{label}-roundtrip-tenant"
    evidence = sample_causal_evidence(
        tenant_id=tenant_id,
        provider="document_store",
        transport_task_id=f"{label}-transport",
    )
    stored = store.append(evidence)
    assert stored == evidence

    by_execution = store.list_for_execution(
        tenant_id=tenant_id,
        task_id=evidence.target.task_id,
        run_id=evidence.target.run_id,
    )
    by_transport = store.list_for_transport_task(
        tenant_id=tenant_id,
        provider=evidence.source.provider,
        transport_task_id=evidence.source.task_id,
    )
    assert by_execution == (evidence,)
    assert by_transport == (evidence,)
    assert by_execution[0].source.provider == evidence.source.provider
    assert by_execution[0].source.task_id == evidence.source.task_id
    assert by_execution[0].target.attempt_id == evidence.target.attempt_id
