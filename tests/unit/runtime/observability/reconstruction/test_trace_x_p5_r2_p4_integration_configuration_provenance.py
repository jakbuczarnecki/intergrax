# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4 integration configuration provenance reconstruction."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.integrations.integration_configuration_provenance_reader import (
    PinningStoreExecutionIntegrationConfigurationProvenanceReader,
)
from intergrax.applications._shared.integrations.persistence import (
    InMemoryExecutionIntegrationConfigurationPinningStore,
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
    validate_execution_id,
)
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceMode,
    ExecutionIntegrationConfigurationProvenanceReadStatus,
    IntegrationConfigurationSubject,
)
from intergrax.contracts.execution_integration_configuration_provenance_requirement import (
    derive_integration_configuration_provenance_requirement_event_id,
)
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationRequirementRecoveryStaging,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningError,
    ExecutionIntegrationConfigurationPinningFailureReason,
)
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads.spine_families import (
    IntegrationConfigurationProvenanceRequirementPayloadV1,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
)
from tests.unit.applications.integrations.test_trace_x_p5_r2_p2_persistence import (
    _EXEC,
    _configured_slice,
    _provenance_configured_adopted,
    _subject,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_TENANT_B = "tenant-b"


def _runtime_event(
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    payload: dict[str, object] | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=mint_event_id(),
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        payload=payload or {},
    )


def _append_event(store: InMemoryRuntimeEventStore, event: RuntimeEvent) -> None:
    store.append(event, tenant_id=event.tenant_id or _TENANT)


def _recovery_staging() -> ExecutionIntegrationConfigurationRequirementRecoveryStaging:
    return ExecutionIntegrationConfigurationRequirementRecoveryStaging(
        requirement_boundary_prepared_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )


def _pin_adopted(
    store: InMemoryExecutionIntegrationConfigurationPinningStore,
    *,
    subject: IntegrationConfigurationSubject,
    provenance: ExecutionIntegrationConfigurationProvenance,
) -> None:
    store.pin(
        subject=subject,
        provenance=provenance,
        requirement_recovery_staging=_recovery_staging(),
    )


def _append_requirement_spine(
    store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    subject: IntegrationConfigurationSubject,
) -> None:
    payload = IntegrationConfigurationProvenanceRequirementPayloadV1(
        integration_category=subject.integration_category.value,
        provider_id=subject.provider_id,
        resource_scope=subject.resource_scope,
        configuration_type=subject.configuration_type,
        provenance_mode=ExecutionIntegrationConfigurationProvenanceMode.CONFIGURED_ADOPTED.value,
    )
    event_id = derive_integration_configuration_provenance_requirement_event_id(
        tenant_id=tenant_id,
        execution_id=execution_id,
        subject=subject,
    )
    event = RuntimeEvent(
        event_id=event_id,
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        event_type=RuntimeEventType.INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        payload={},
    )
    _append_event(store, runtime_event_with_payload(event, payload))


def test_integration_reader_absent_yields_not_configured() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    reconstructor = ExecutionReconstructor(
        InMemoryRuntimeEventStore(),
        InMemoryCausalEvidencePersistence(),
    )
    view = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert view.execution_integration_configuration_provenance == ()
    assert (
        view.integration_configuration_provenance_read_status
        is ExecutionIntegrationConfigurationProvenanceReadStatus.NOT_CONFIGURED
    )


def test_reader_present_without_pins_yields_configured_empty() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=execution_id,
        ),
    )
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(
        InMemoryExecutionIntegrationConfigurationPinningStore(),
    )
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=reader,
    )
    view = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert view.execution_integration_configuration_provenance == ()
    assert (
        view.integration_configuration_provenance_read_status
        is ExecutionIntegrationConfigurationProvenanceReadStatus.CONFIGURED
    )


def test_configured_execution_reconstructs_persisted_provenance() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    execution_id = _EXEC
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    record = _provenance_configured_adopted()
    _pin_adopted(pinning, subject=_subject(), provenance=record)
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=execution_id,
        ),
    )
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=reader,
    )
    view = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert view.execution_integration_configuration_provenance == (record,)


def test_multiple_subjects_preserved_in_store_order() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    first = _provenance_configured_adopted()
    second = ExecutionIntegrationConfigurationProvenance(
        tenant_id=first.tenant_id,
        execution_id=first.execution_id,
        mode=first.mode,
        effective=first.effective,
        configured=_configured_slice(resource_scope="scope-b"),
    )
    _pin_adopted(pinning, subject=_subject(resource_scope="scope-a"), provenance=first)
    _pin_adopted(pinning, subject=_subject(resource_scope="scope-b"), provenance=second)
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=_EXEC,
        ),
    )
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=reader,
    )
    view = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert len(view.execution_integration_configuration_provenance) == 2


def test_tenant_mismatch_in_record_fails_closed() -> None:
    bad = _provenance_configured_adopted()

    class _CorruptReader:
        def read_all(
            self,
            *,
            tenant_id: str,
            execution_id: ExecutionId,
        ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
            corrupt = ExecutionIntegrationConfigurationProvenance(
                tenant_id=_TENANT_B,
                execution_id=execution_id,
                mode=bad.mode,
                effective=bad.effective,
                configured=_configured_slice(tenant_id=_TENANT_B),
            )
            return (corrupt,)

    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=_EXEC,
        ),
    )
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=_CorruptReader(),
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT, task_id, run_id)


def test_execution_id_mismatch_fails_closed() -> None:
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    record = _provenance_configured_adopted()
    _pin_adopted(pinning, subject=_subject(), provenance=record)
    other_execution = mint_execution_id()

    class _CorruptReader:
        def read_all(
            self,
            *,
            tenant_id: str,
            execution_id: ExecutionId,
        ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
            corrupt = ExecutionIntegrationConfigurationProvenance(
                tenant_id=record.tenant_id,
                execution_id=other_execution,
                mode=record.mode,
                effective=record.effective,
                configured=record.configured,
            )
            return (corrupt,)

    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=_EXEC,
        ),
    )
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=_CorruptReader(),
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT, task_id, run_id)


def test_corrupt_pinning_store_fails_closed() -> None:
    class _FailingReader:
        def read_all(
            self,
            *,
            tenant_id: str,
            execution_id: ExecutionId,
        ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
            raise ExecutionIntegrationConfigurationPinningError(
                ExecutionIntegrationConfigurationPinningFailureReason.CORRUPT_RECORD,
            )

    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=_EXEC,
        ),
    )
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=_FailingReader(),
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT, task_id, run_id)


def test_required_provenance_missing_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    attempt_id = mint_attempt_id()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=_EXEC,
        ),
    )
    _append_requirement_spine(
        store,
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=_EXEC,
        subject=_subject(),
    )
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(
        InMemoryExecutionIntegrationConfigurationPinningStore(),
    )
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=reader,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError, match="required"):
        reconstructor.reconstruct_execution(_TENANT, task_id, run_id)


def test_child_execution_does_not_inherit_parent_provenance() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    parent_execution = _EXEC
    child_execution = mint_execution_id()
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    _pin_adopted(pinning, subject=_subject(), provenance=_provenance_configured_adopted())
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=parent_execution,
        ),
    )
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=child_execution,
        ),
    )
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=reader,
    )
    view = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert len(view.execution_integration_configuration_provenance) == 1
    assert view.execution_integration_configuration_provenance[0].execution_id == parent_execution


def test_historical_restart_ignores_changed_current_configuration_state() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    historical = _provenance_configured_adopted()
    _pin_adopted(pinning, subject=_subject(), provenance=historical)
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=_EXEC,
        ),
    )
    reader_after_restart = PinningStoreExecutionIntegrationConfigurationProvenanceReader(
        pinning,
    )
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=reader_after_restart,
    )
    view = reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    assert view.execution_integration_configuration_provenance[0].configured is not None
    assert (
        view.execution_integration_configuration_provenance[0].configured.configuration_fingerprint
        == "fp-test-001"
    )
    restarted = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=(
            PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
        ),
    )
    view_again = restarted.reconstruct_execution(_TENANT, task_id, run_id)
    assert (
        view_again.execution_integration_configuration_provenance
        == view.execution_integration_configuration_provenance
    )


def test_requirement_spine_malformed_payload_reconstruction_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    _pin_adopted(pinning, subject=_subject(), provenance=_provenance_configured_adopted())
    store = InMemoryRuntimeEventStore()
    event_id = derive_integration_configuration_provenance_requirement_event_id(
        tenant_id=_TENANT,
        execution_id=_EXEC,
        subject=_subject(),
    )
    malformed = RuntimeEvent(
        event_id=event_id,
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=_EXEC,
        event_type=RuntimeEventType.INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        payload={
            "payload_schema_id": "integration_configuration_provenance_requirement.unknown.v99",
            "payload": {"integration_category": "invalid"},
        },
    )
    _append_event(store, malformed)
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=_EXEC,
        ),
    )
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=reader,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError, match="requirement payload"):
        reconstructor.reconstruct_execution(_TENANT, task_id, run_id)


def test_reconstruction_does_not_resolve_providers() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    pinning = InMemoryExecutionIntegrationConfigurationPinningStore()
    _pin_adopted(pinning, subject=_subject(), provenance=_provenance_configured_adopted())
    store = InMemoryRuntimeEventStore()
    _append_event(
        store,
        _runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=_EXEC,
        ),
    )
    reader = PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_integration_configuration_provenance_reader=reader,
    )
    resolve_mock = MagicMock()
    reconstructor.reconstruct_execution(_TENANT, task_id, run_id)
    resolve_mock.assert_not_called()
