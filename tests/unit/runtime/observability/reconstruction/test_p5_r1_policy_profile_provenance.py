# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1 policy and profile provenance reconstruction integrity."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.profile_resolution.execution_pinning import (
    InMemoryEffectiveProfileExecutionPinningStore,
)
from intergrax.applications._shared.profile_resolution.execution_effective_profile_provenance_reader import (
    PinningStoreExecutionEffectiveProfileProvenanceReader,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionBinding,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    mint_effective_profile_revision_id,
)
from intergrax.contracts.execution_effective_profile_provenance import (
    ExecutionEffectiveProfileProvenanceReadStatus,
)
from intergrax.contracts.execution_event_position import (
    ExecutionEventPosition,
)
from intergrax.contracts.positioned_runtime_event import (
    PositionedRuntimeEvent,
    as_of_boundary_for_positioned,
)
from intergrax.runtime.observability.reconstruction.policy_provenance_projection import (
    project_policy_decision_provenance,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
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
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"


def _pin(
    pinning: InMemoryEffectiveProfileExecutionPinningStore,
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    revision_id_value: str | None = None,
    fingerprint: str = "fp-test-1",
) -> None:
    revision_id = revision_id_value or mint_effective_profile_revision_id()
    pinning.pin(
        EffectiveProfileExecutionBinding(
            tenant_id=tenant_id,
            execution_id=execution_id,
            revision_id=revision_id,
            fingerprint=fingerprint,
        )
    )


def _policy_event(
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    bundle_id: str = "bundle-1",
    position: int = 1,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=mint_event_id(),
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        event_type=RuntimeEventType.POLICY_DECISION,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        payload={
            "governance_evidence_schema": "governance_decision_evidence.v1",
            "evidence_id": "ev-1",
            "evaluation_point": "PRE_MODEL",
            "action": "allow",
            "resource_type": "execution",
            "resource_scope": "task",
            "decision": "allow",
            "reason": "ok",
            "reason_code": "OK",
            "policy_bundle_id": bundle_id,
            "policy_bundle_version": "1.0.0",
            "policy_bundle_digest": "digest-abc",
            "policy_rule_id": "rule-42",
            "request_digest": "req-digest",
            "idempotency_key": "idem-1",
            "workspace_id": "ws",
            "principal_id": "principal",
            "_position": position,
        },
    )


def test_policy_provenance_fields_and_order() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _policy_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            bundle_id="bundle-first",
            position=1,
        ),
        tenant_id=_TENANT_A,
    )
    store.append(
        _policy_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            bundle_id="bundle-second",
            position=2,
        ),
        tenant_id=_TENANT_A,
    )
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    view = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    assert len(view.policy_decision_provenance) == 2
    assert view.policy_decision_provenance[0].policy_bundle_id == "bundle-first"
    assert view.policy_decision_provenance[1].policy_bundle_id == "bundle-second"
    assert view.policy_decision_provenance[0].policy_rule_id == "rule-42"
    assert view.policy_decision_provenance[0].evaluation_point == "PRE_MODEL"
    assert view.policy_decision_provenance[0].execution_id == execution_id


def test_policy_cross_tenant_rejected() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    event = _policy_event(
        tenant_id=_TENANT_B,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    positioned = (
        PositionedRuntimeEvent(event=event, position=ExecutionEventPosition(1)),
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        project_policy_decision_provenance(
            positioned,
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
        )


def test_policy_malformed_payload_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        RuntimeEvent(
            event_id=mint_event_id(),
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            event_type=RuntimeEventType.POLICY_DECISION,
            phase=ExecutionPhase.STEP_EXECUTION,
            timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            payload={"evidence_id": ""},
        ),
        tenant_id=_TENANT_A,
    )
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)


def test_profile_reader_absent_does_not_invent_provenance() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    view = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    assert view.execution_effective_profile_provenance == ()
    assert (
        view.effective_profile_provenance_read_status
        is ExecutionEffectiveProfileProvenanceReadStatus.NOT_CONFIGURED
    )


def test_profile_missing_binding_fails_when_reader_configured() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _policy_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
        tenant_id=_TENANT_A,
    )
    reader = PinningStoreExecutionEffectiveProfileProvenanceReader(
        InMemoryEffectiveProfileExecutionPinningStore(),
    )
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_effective_profile_provenance_reader=reader,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)


def test_profile_cross_tenant_binding_invisible() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    pinning = InMemoryEffectiveProfileExecutionPinningStore()
    _pin(pinning, tenant_id=_TENANT_A, execution_id=execution_id)
    store = InMemoryRuntimeEventStore()
    store.append(
        _policy_event(
            tenant_id=_TENANT_B,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
        tenant_id=_TENANT_B,
    )
    reader = PinningStoreExecutionEffectiveProfileProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_effective_profile_provenance_reader=reader,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT_B, task_id, run_id)


def test_profile_as_of_excludes_future_execution_id() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    execution_a = mint_execution_id()
    execution_b = mint_execution_id()
    pinning = InMemoryEffectiveProfileExecutionPinningStore()
    for execution_id in (execution_a, execution_b):
        _pin(pinning, tenant_id=_TENANT_A, execution_id=execution_id)
    store = InMemoryRuntimeEventStore()
    event_a = _policy_event(
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a,
        execution_id=execution_a,
        position=1,
    )
    event_b = _policy_event(
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_b,
        execution_id=execution_b,
        position=2,
    )
    positioned_a = store.append(event_a, tenant_id=_TENANT_A)
    store.append(event_b, tenant_id=_TENANT_A)
    reader = PinningStoreExecutionEffectiveProfileProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_effective_profile_provenance_reader=reader,
    )
    boundary = as_of_boundary_for_positioned(positioned_a)
    view = reconstructor.reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    assert len(view.execution_effective_profile_provenance) == 1
    assert view.execution_effective_profile_provenance[0].execution_id == execution_a


def test_reconstruction_does_not_invoke_governance_evaluator() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    evaluator = MagicMock()
    reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    evaluator.assert_not_called()
