# © Artur Czarnecki. All rights reserved.

"""R2 execution failure evidence qualification tests."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_failure_evidence import ExecutionFailureKind
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.events.evidence_durability import (
    EvidencePersistenceRequirement,
    evidence_persistence_requirement,
)
from intergrax.runtime.events.event_catalog import get_catalog_entry
from intergrax.runtime.events.payloads.canonical import ExecutionFailurePayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_taxonomy import EventCategory, RetentionClass


def test_execution_failed_catalog_registration() -> None:
    entry = get_catalog_entry(RuntimeEventType.EXECUTION_FAILED)
    assert entry is not None
    assert entry.phase is ExecutionPhase.STEP_EXECUTION
    assert entry.ops_hint == "ops:alert"
    assert entry.category is EventCategory.PLATFORM
    assert entry.retention_class is RetentionClass.OPERATIONAL
    assert entry.sample_rate == 1.0
    assert entry.preferred_payload_schema_id == "execution_failure.v1"


def test_execution_failed_mandatory_persistence() -> None:
    event = RuntimeEvent(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        event_type=RuntimeEventType.EXECUTION_FAILED,
        phase=ExecutionPhase.STEP_EXECUTION,
    )
    assert (
        evidence_persistence_requirement(event)
        is EvidencePersistenceRequirement.MANDATORY
    )


def test_execution_failure_payload_rejects_extra_field() -> None:
    with pytest.raises(Exception):
        ExecutionFailurePayloadV1(
            failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
            safe_summary="Execution delegate failed",
            extra_field="nope",
        )


def test_execution_failure_payload_rejects_secret_in_summary_size() -> None:
    with pytest.raises(Exception):
        ExecutionFailurePayloadV1(
            failure_kind=ExecutionFailureKind.DELEGATE_EXCEPTION,
            safe_summary="x" * 300,
        )


@pytest.mark.asyncio
async def test_cancellation_does_not_emit_execution_failed() -> None:
    from intergrax.runtime.execution.boundary import ExecutionBoundary
    from intergrax.runtime.execution.failure_evidence.active_context import (
        ActiveExecutionEvidenceContext,
        bind_active_execution_evidence_context,
        reset_active_execution_evidence_context,
    )
    from intergrax.runtime.execution.failure_evidence.recording_delegate import (
        ExecutionFailureRecordingDelegate,
    )
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
        reset_active_execution_identity,
    )
    from intergrax.runtime.events.stores.memory_runtime_event_store import (
        InMemoryRuntimeEventStore,
    )
    from intergrax.runtime.events.event_bus import RuntimeEventBus
    from intergrax.runtime.execution.failure_evidence.runtime_event_recorder import (
        RuntimeEventExecutionFailureEvidenceRecorder,
    )

    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    recorder = RuntimeEventExecutionFailureEvidenceRecorder(bus)
    tenant = "tenant-cancel"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()

    class _CancelDelegate:
        async def execute(self, request: object) -> object:
            raise asyncio.CancelledError()

    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    evidence_token = bind_active_execution_evidence_context(
        ActiveExecutionEvidenceContext(
            tenant_id=tenant,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            recorder=recorder,
        ),
    )
    boundary = ExecutionBoundary[object, object](
        ExecutionFailureRecordingDelegate(_CancelDelegate()),
        identity=None,
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await boundary.execute(object())
    finally:
        reset_active_execution_evidence_context(evidence_token)
        reset_active_execution_identity(identity_token)

    events = store.list_for_task(str(task_id), tenant_id=tenant, limit=20)
    assert not any(e.event_type is RuntimeEventType.EXECUTION_FAILED for e in events)
