# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Operator recovery command tests for durable reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import KnowledgeSyncCorruptState
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationRecoveryCommand,
    KnowledgeReconciliationRecoveryCommandKind,
    KnowledgeReconciliationRunAborted,
    KnowledgeReconciliationRunCollecting,
    KnowledgeReconciliationRunPagePrepared,
    KnowledgeReconciliationRunPhase,
    KnowledgeReconciliationRunRecoveryRequired,
    canonical_prepared_state_mutations_fingerprint,
    knowledge_cursor_fingerprint_sha256,
    recovery_evidence_from_run,
)
from tests.unit.runtime.vendor_knowledge.test_reconciliation_durable_coordinator import (
    _durable_coordinator,
)

_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
_DELIVERY = "a" * 64
_FINGERPRINT = "b" * 64
_NULL_CURSOR_FP = knowledge_cursor_fingerprint_sha256(None)


def _collecting() -> KnowledgeReconciliationRunCollecting:
    return KnowledgeReconciliationRunCollecting(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=1,
        created_at=_NOW,
        updated_at=_NOW,
        current_input_cursor_fingerprint=_NULL_CURSOR_FP,
        remaining_candidate_remote_ids=(),
    )


def _page_prepared() -> KnowledgeReconciliationRunPagePrepared:
    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeReconciliationPreparedStateMutationTemplate,
        KnowledgeRemoteItemStatus,
    )
    from intergrax.runtime.vendor_knowledge.models import KnowledgeItemRevision

    templates = (
        KnowledgeReconciliationPreparedStateMutationTemplate(
            remote_id="item-1",
            resulting_status=KnowledgeRemoteItemStatus.ACTIVE,
            revision=KnowledgeItemRevision(version="1"),
            binding_configuration_version=1,
        ),
    )
    return KnowledgeReconciliationRunPagePrepared(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=2,
        created_at=_NOW,
        updated_at=_NOW,
        prepared_input_cursor_fingerprint=_NULL_CURSOR_FP,
        provider_page_fingerprint=_FINGERPRINT,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
        prepared_state_mutation_templates=templates,
        prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
            templates
        ),
        prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
        prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
        prepared_page_size=100,
        has_more=False,
        delivery_id=_DELIVERY,
        remaining_candidate_remote_ids=(),
        synthetic_tombstone_remote_ids=(),
    )


@pytest.mark.unit
def test_abort_pristine_collecting_succeeds() -> None:
    coordinator, _, _, checkpoint, runs, _, _ = _durable_coordinator()
    collecting = _collecting()
    runs.runs[("tenant-1", "binding-1")] = collecting
    command = KnowledgeReconciliationRecoveryCommand(
        kind=KnowledgeReconciliationRecoveryCommandKind.ABORT_PRISTINE,
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_run_id="run-1",
        expected_run_record_version=1,
        expected_phase=KnowledgeReconciliationRunPhase.COLLECTING,
        operator_reason_code="abort_pristine",
    )
    coordinator.execute_reconciliation_recovery(command)
    aborted = runs.runs[("tenant-1", "binding-1")]
    assert isinstance(aborted, KnowledgeReconciliationRunAborted)
    assert aborted.operator_reason_code == "abort_pristine"
    assert checkpoint.commit_calls == []


@pytest.mark.unit
def test_abort_pristine_page_prepared_both_absent_succeeds() -> None:
    coordinator, _, _, _, runs, _, inspector = _durable_coordinator()
    prepared = _page_prepared()
    runs.runs[("tenant-1", "binding-1")] = prepared
    command = KnowledgeReconciliationRecoveryCommand(
        kind=KnowledgeReconciliationRecoveryCommandKind.ABORT_PRISTINE,
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_run_id="run-1",
        expected_run_record_version=2,
        expected_phase=KnowledgeReconciliationRunPhase.PAGE_PREPARED,
        operator_reason_code="abort_pristine",
    )
    coordinator.execute_reconciliation_recovery(command)
    assert (
        runs.runs[("tenant-1", "binding-1")].phase
        is KnowledgeReconciliationRunPhase.ABORTED
    )
    assert inspector.durable == {}


@pytest.mark.unit
def test_abort_pristine_rejects_sink_applied() -> None:
    coordinator, _, _, _, runs, _, inspector = _durable_coordinator()
    prepared = _page_prepared()
    runs.runs[("tenant-1", "binding-1")] = prepared
    inspector.durable[_DELIVERY] = _FINGERPRINT
    command = KnowledgeReconciliationRecoveryCommand(
        kind=KnowledgeReconciliationRecoveryCommandKind.ABORT_PRISTINE,
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_run_id="run-1",
        expected_run_record_version=2,
        expected_phase=KnowledgeReconciliationRunPhase.PAGE_PREPARED,
        operator_reason_code="abort_pristine",
    )
    with pytest.raises(VendorKnowledgeError):
        coordinator.execute_reconciliation_recovery(command)


@pytest.mark.unit
def test_repair_required_preserves_recovery_required() -> None:
    coordinator, _, _, _, runs, _, _ = _durable_coordinator()
    collecting = _collecting()
    recovery = KnowledgeReconciliationRunRecoveryRequired(
        **collecting.model_dump(
            exclude={
                "phase",
                "record_version",
                "updated_at",
                "candidate_inventory_continuation_token",
                "current_input_cursor",
                "current_input_cursor_fingerprint",
                "remaining_candidate_remote_ids",
            }
        ),
        phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        record_version=2,
        updated_at=_NOW,
        recovery_reason_code="provider_page_mismatch",
        recovery_evidence=recovery_evidence_from_run(collecting),
    )
    runs.runs[("tenant-1", "binding-1")] = recovery
    command = KnowledgeReconciliationRecoveryCommand(
        kind=KnowledgeReconciliationRecoveryCommandKind.REPAIR_REQUIRED,
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_run_id="run-1",
        expected_run_record_version=2,
        expected_phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        operator_reason_code="manual_repair",
    )
    coordinator.execute_reconciliation_recovery(command)
    updated = runs.runs[("tenant-1", "binding-1")]
    assert isinstance(updated, KnowledgeReconciliationRunRecoveryRequired)
    assert updated.recovery_reason_code == "manual_repair"


@pytest.mark.unit
def test_generic_cas_replace_recovery_exit_blocked() -> None:
    coordinator, _, _, _, runs, _, _ = _durable_coordinator()
    collecting = _collecting()
    recovery = KnowledgeReconciliationRunRecoveryRequired(
        **collecting.model_dump(
            exclude={
                "phase",
                "record_version",
                "updated_at",
                "candidate_inventory_continuation_token",
                "current_input_cursor",
                "current_input_cursor_fingerprint",
                "remaining_candidate_remote_ids",
            }
        ),
        phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        record_version=2,
        updated_at=_NOW,
        recovery_reason_code="provider_page_mismatch",
        recovery_evidence=recovery_evidence_from_run(collecting),
    )
    runs.runs[("tenant-1", "binding-1")] = recovery
    with pytest.raises(KnowledgeSyncCorruptState):
        runs.cas_replace(
            expected=recovery,
            replacement=_collecting().model_copy(update={"record_version": 3}),
        )


@pytest.mark.unit
def test_stale_recovery_command_rejected() -> None:
    coordinator, _, _, _, runs, _, _ = _durable_coordinator()
    runs.runs[("tenant-1", "binding-1")] = _collecting()
    command = KnowledgeReconciliationRecoveryCommand(
        kind=KnowledgeReconciliationRecoveryCommandKind.ABORT_PRISTINE,
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_run_id="run-1",
        expected_run_record_version=99,
        expected_phase=KnowledgeReconciliationRunPhase.COLLECTING,
        operator_reason_code="abort_pristine",
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        coordinator.execute_reconciliation_recovery(command)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert "secret" not in exc_info.value.safe_message.lower()


@pytest.mark.unit
def test_sync_resume_exact_rejects_page_prepared_evidence() -> None:
    coordinator, _, _, _, runs, _, _ = _durable_coordinator()
    prepared = _page_prepared()
    recovery = KnowledgeReconciliationRunRecoveryRequired(
        **prepared.model_dump(
            exclude={
                "phase",
                "record_version",
                "updated_at",
                "prepared_input_cursor",
                "prepared_input_cursor_fingerprint",
                "provider_page_fingerprint",
                "prepared_batch_payload_fingerprint",
                "prepared_state_mutation_templates",
                "prepared_state_mutations_fingerprint",
                "prepared_proposed_checkpoint",
                "prepared_proposed_checkpoint_fingerprint",
                "prepared_next_cursor",
                "prepared_next_cursor_fingerprint",
                "prepared_page_size",
                "delivery_id",
                "has_more",
                "synthetic_tombstone_remote_ids",
                "prepared_parent_delivery_id",
                "remaining_candidate_remote_ids",
            }
        ),
        phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        record_version=3,
        updated_at=_NOW,
        recovery_reason_code="provider_page_mismatch",
        recovery_evidence=recovery_evidence_from_run(prepared),
    )
    runs.runs[("tenant-1", "binding-1")] = recovery
    command = KnowledgeReconciliationRecoveryCommand(
        kind=KnowledgeReconciliationRecoveryCommandKind.RESUME_EXACT,
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_run_id="run-1",
        expected_run_record_version=3,
        expected_phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        operator_reason_code="resume_exact",
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        coordinator.execute_reconciliation_recovery(command)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert "async" in exc_info.value.safe_message.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_page_prepared_corrupt_item_receipt_enters_recovery() -> None:
    from dataclasses import dataclass

    from intergrax.runtime.vendor_knowledge.sync_contracts import (
        KnowledgeSyncCorruptState,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )
    from tests.unit.runtime.vendor_knowledge._sync_fakes import (
        InMemoryRemoteItemStateRepository,
    )

    @dataclass
    class _CorruptItemInspector(InMemoryRemoteItemStateRepository):
        inspect_calls: int = 0

        def inspect_delivery_receipt(self, **kwargs):
            self.inspect_calls += 1
            raise KnowledgeSyncCorruptState("corrupt item receipt")

    operation_id = "op-corrupt-item"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    state = _CorruptItemInspector()
    coordinator, _, _, _, runs, _, inspector = _durable_coordinator(state=state)
    prepared = _page_prepared().model_copy(update={"run_id": run_id})
    runs.runs[("tenant-1", "binding-1")] = prepared
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    updated = runs.runs[("tenant-1", "binding-1")]
    assert updated.phase is KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED
    assert state.inspect_calls == 1
    assert inspector.durable == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_page_prepared_corrupt_sink_receipt_enters_recovery() -> None:
    from dataclasses import dataclass

    from intergrax.runtime.vendor_knowledge.sync_contracts import (
        KnowledgeSyncCorruptState,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )
    from tests.unit.runtime.vendor_knowledge._sync_fakes import (
        RecordingSinkReceiptInspector,
    )

    @dataclass
    class _CountingSinkInspector(RecordingSinkReceiptInspector):
        inspect_calls: int = 0

        def inspect_receipt(self, **kwargs):
            self.inspect_calls += 1
            raise KnowledgeSyncCorruptState("corrupt sink receipt")

    operation_id = "op-corrupt-sink"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    inspector = _CountingSinkInspector()
    coordinator, _, _, _, runs, _, _ = _durable_coordinator()
    coordinator._sink_receipt_inspector = inspector  # type: ignore[attr-defined]
    coordinator._reconciliation_engine._sink_receipt_inspector = inspector  # type: ignore[attr-defined]
    prepared = _page_prepared().model_copy(update={"run_id": run_id})
    runs.runs[("tenant-1", "binding-1")] = prepared
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert runs.runs[("tenant-1", "binding-1")].phase is (
        KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED
    )
    assert inspector.inspect_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_page_prepared_receipt_backend_failure_preserves_phase() -> None:
    from dataclasses import dataclass

    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )
    from tests.unit.runtime.vendor_knowledge._sync_fakes import (
        InMemoryRemoteItemStateRepository,
    )

    @dataclass
    class _FailingItemInspector(InMemoryRemoteItemStateRepository):
        inspect_calls: int = 0

        def inspect_delivery_receipt(self, **kwargs):
            self.inspect_calls += 1
            raise RuntimeError("backend down")

    operation_id = "op-receipt-backend"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    state = _FailingItemInspector()
    coordinator, _, _, _, runs, _, _ = _durable_coordinator(state=state)
    prepared = _page_prepared().model_copy(update={"run_id": run_id})
    runs.runs[("tenant-1", "binding-1")] = prepared
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert runs.runs[("tenant-1", "binding-1")].phase is (
        KnowledgeReconciliationRunPhase.PAGE_PREPARED
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_resume_item_completed_inspects_sink_zero_times() -> None:
    from dataclasses import dataclass, field

    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeRemoteItemStateReceipt,
        KnowledgeRemoteItemStateReceiptStatus,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )
    from tests.unit.runtime.vendor_knowledge._sync_fakes import (
        InMemoryRemoteItemStateRepository,
        RecordingSinkReceiptInspector,
    )

    @dataclass
    class _CompletedItemInspector(InMemoryRemoteItemStateRepository):
        inspect_calls: int = 0

        def inspect_delivery_receipt(self, **kwargs):
            self.inspect_calls += 1
            return KnowledgeRemoteItemStateReceipt(
                status=KnowledgeRemoteItemStateReceiptStatus.COMPLETED,
                delivery_id=kwargs["delivery_id"],
                prepared_state_mutations_fingerprint=kwargs[
                    "prepared_state_mutations_fingerprint"
                ],
            )

    @dataclass
    class _FailSinkInspector(RecordingSinkReceiptInspector):
        inspect_calls: int = 0

        def inspect_receipt(self, **kwargs):
            self.inspect_calls += 1
            raise AssertionError("sink receipt must not be inspected")

    operation_id = "op-resume-completed"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    state = _CompletedItemInspector()
    inspector = _FailSinkInspector()
    coordinator, _, _, _, runs, _, _ = _durable_coordinator(state=state)
    coordinator._sink_receipt_inspector = inspector  # type: ignore[attr-defined]
    coordinator._reconciliation_engine._sink_receipt_inspector = inspector  # type: ignore[attr-defined]
    prepared = _page_prepared().model_copy(update={"run_id": run_id})
    recovery = KnowledgeReconciliationRunRecoveryRequired(
        **prepared.model_dump(
            exclude={
                "phase",
                "record_version",
                "updated_at",
                "prepared_input_cursor",
                "prepared_input_cursor_fingerprint",
                "provider_page_fingerprint",
                "prepared_batch_payload_fingerprint",
                "prepared_state_mutation_templates",
                "prepared_state_mutations_fingerprint",
                "prepared_proposed_checkpoint",
                "prepared_proposed_checkpoint_fingerprint",
                "prepared_next_cursor",
                "prepared_next_cursor_fingerprint",
                "prepared_page_size",
                "delivery_id",
                "has_more",
                "synthetic_tombstone_remote_ids",
                "prepared_parent_delivery_id",
                "remaining_candidate_remote_ids",
            }
        ),
        phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        record_version=3,
        updated_at=_NOW,
        recovery_reason_code="provider_page_mismatch",
        recovery_evidence=recovery_evidence_from_run(prepared),
    )
    runs.runs[("tenant-1", "binding-1")] = recovery
    command = KnowledgeReconciliationRecoveryCommand(
        kind=KnowledgeReconciliationRecoveryCommandKind.RESUME_EXACT,
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_run_id=run_id,
        expected_run_record_version=3,
        expected_phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
        operator_reason_code="resume_exact",
    )
    result = await coordinator.execute_reconciliation_recovery_async(command)
    assert result.phase is KnowledgeReconciliationRunPhase.PAGE_PREPARED
    assert state.inspect_calls == 1
    assert inspector.inspect_calls == 0
