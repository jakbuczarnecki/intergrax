# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for reconciliation-finalization models and receipt contracts."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeCursor,
    KnowledgeItemRevision,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeSyncSinkReceiptInspector,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationLimitPolicy,
    KnowledgeReconciliationPreparedStateMutationTemplate,
    KnowledgeReconciliationRecoveryCommand,
    KnowledgeReconciliationRecoveryCommandKind,
    KnowledgeReconciliationRunAborted,
    KnowledgeReconciliationRunCollecting,
    KnowledgeReconciliationRunCompleted,
    KnowledgeReconciliationRunFinalizing,
    KnowledgeReconciliationRunPagePrepared,
    KnowledgeReconciliationRunPhase,
    KnowledgeReconciliationRunRecoveryRequired,
    KnowledgeRemoteItemStateReceipt,
    KnowledgeRemoteItemStateReceiptStatus,
    KnowledgeRemoteItemStatus,
    KnowledgeSyncCheckpoint,
    KnowledgeSyncSinkReceipt,
    KnowledgeSyncSinkReceiptStatus,
    canonical_prepared_state_mutations_fingerprint,
    knowledge_cursor_fingerprint_sha256,
    knowledge_sync_checkpoint_fingerprint_sha256,
    parse_knowledge_reconciliation_run,
    validate_reconciliation_candidate_inventory,
    validate_reconciliation_prepared_intent,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    RecordingSinkReceiptInspector,
)

_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
_DELIVERY = "a" * 64
_FINGERPRINT = "b" * 64
_NULL_CURSOR_FP = knowledge_cursor_fingerprint_sha256(None)
_CURSOR = KnowledgeCursor(
    value="https://example.test/continue?token=secret", version="v1"
)
_CURSOR_FP = knowledge_cursor_fingerprint_sha256(_CURSOR)


def _checkpoint(*, cursor_value: str = "cursor-1") -> KnowledgeSyncCheckpoint:
    return KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value=cursor_value, version="v1"),
    )


def _collecting(
    *,
    applied_page_count: int = 0,
    last_applied_delivery_id: str | None = None,
) -> KnowledgeReconciliationRunCollecting:
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
        applied_page_count=applied_page_count,
        last_applied_delivery_id=last_applied_delivery_id,
        current_input_cursor_fingerprint=_NULL_CURSOR_FP,
        remaining_candidate_remote_ids=("item-b", "item-a"),
    )


def _template(
    remote_id: str = "item-1",
) -> KnowledgeReconciliationPreparedStateMutationTemplate:
    return KnowledgeReconciliationPreparedStateMutationTemplate(
        remote_id=remote_id,
        resulting_status=KnowledgeRemoteItemStatus.ACTIVE,
        revision=KnowledgeItemRevision(version="1"),
        binding_configuration_version=1,
    )


def _page_prepared(*, has_more: bool = True) -> KnowledgeReconciliationRunPagePrepared:
    templates = (_template("item-a"), _template("item-b"))
    mutations_fp = canonical_prepared_state_mutations_fingerprint(templates)
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
        applied_page_count=0,
        prepared_input_cursor_fingerprint=_NULL_CURSOR_FP,
        provider_page_fingerprint=_FINGERPRINT,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
        prepared_state_mutation_templates=templates,
        prepared_state_mutations_fingerprint=mutations_fp,
        prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
        prepared_next_cursor_fingerprint=_CURSOR_FP,
        prepared_next_cursor=_CURSOR,
        has_more=has_more,
        delivery_id=_DELIVERY,
        remaining_candidate_remote_ids=("item-c",),
        synthetic_tombstone_remote_ids=() if has_more else ("item-z",),
    )


@pytest.mark.unit
def test_all_reconciliation_phases_validate() -> None:
    collecting = _collecting()
    assert collecting.phase is KnowledgeReconciliationRunPhase.COLLECTING
    assert collecting.effects_started is False
    prepared = _page_prepared()
    assert prepared.phase is KnowledgeReconciliationRunPhase.PAGE_PREPARED
    checkpoint = _checkpoint()
    finalizing = KnowledgeReconciliationRunFinalizing(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=3,
        created_at=_NOW,
        updated_at=_NOW,
        applied_page_count=1,
        last_applied_delivery_id=_DELIVERY,
        intended_final_completed_checkpoint=checkpoint,
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            checkpoint
        ),
        expected_previous_completed_checkpoint=checkpoint,
        final_delivery_id=_DELIVERY,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
    )
    completed = KnowledgeReconciliationRunCompleted(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=4,
        created_at=_NOW,
        updated_at=_NOW,
        applied_page_count=1,
        last_applied_delivery_id=_DELIVERY,
        committed_completed_checkpoint=checkpoint,
        final_delivery_id=_DELIVERY,
    )
    recovery = KnowledgeReconciliationRunRecoveryRequired(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=5,
        created_at=_NOW,
        updated_at=_NOW,
        applied_page_count=1,
        last_applied_delivery_id=_DELIVERY,
        recovery_reason_code="provider_page_mismatch",
    )
    aborted = KnowledgeReconciliationRunAborted(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=6,
        created_at=_NOW,
        updated_at=_NOW,
        applied_page_count=0,
        operator_reason_code="operator_abort",
    )
    assert finalizing.phase is KnowledgeReconciliationRunPhase.FINALIZING
    assert completed.phase is KnowledgeReconciliationRunPhase.COMPLETED
    assert recovery.phase is KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED
    assert aborted.phase is KnowledgeReconciliationRunPhase.ABORTED


@pytest.mark.unit
def test_unknown_phase_and_field_rejected() -> None:
    payload = _collecting().model_dump(mode="json")
    payload["phase"] = "not-a-phase"
    with pytest.raises(ValueError, match="unknown"):
        parse_knowledge_reconciliation_run(payload)
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunCollecting(
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
            extra_field="nope",  # type: ignore[call-arg]
        )


@pytest.mark.unit
def test_malformed_hashes_and_delivery_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        _collecting(last_applied_delivery_id="not-a-hash")
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunPagePrepared(
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
            prepared_state_mutation_templates=(_template(),),
            prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
                (_template(),)
            ),
            prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
            prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
            has_more=True,
            delivery_id="short",
            remaining_candidate_remote_ids=("item-c",),
        )


@pytest.mark.unit
def test_cursor_fingerprint_pair_and_utc_timestamps() -> None:
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunCollecting(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            provider_id="example",
            source_kind="issues",
            run_id="run-1",
            record_version=1,
            created_at=datetime(2026, 1, 1),
            updated_at=_NOW,
            current_input_cursor_fingerprint=_NULL_CURSOR_FP,
        )
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunCollecting(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            provider_id="example",
            source_kind="issues",
            run_id="run-1",
            record_version=1,
            created_at=_NOW,
            updated_at=_NOW,
            current_input_cursor=_CURSOR,
            current_input_cursor_fingerprint="d" * 64,
        )


@pytest.mark.unit
def test_applied_page_count_and_last_delivery_consistency() -> None:
    with pytest.raises(ValidationError):
        _collecting(applied_page_count=1, last_applied_delivery_id=None)
    with pytest.raises(ValidationError):
        _collecting(applied_page_count=0, last_applied_delivery_id=_DELIVERY)
    prior = _collecting(applied_page_count=2, last_applied_delivery_id=_DELIVERY)
    assert prior.effects_started is True


@pytest.mark.unit
def test_prepared_fields_forbidden_outside_page_prepared() -> None:
    payload = _collecting().model_dump(mode="json")
    payload["delivery_id"] = _DELIVERY
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunCollecting.model_validate(payload)


@pytest.mark.unit
def test_finalizing_fields_forbidden_outside_finalizing() -> None:
    payload = _page_prepared().model_dump(mode="json")
    payload["final_delivery_id"] = _DELIVERY
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunPagePrepared.model_validate(payload)


@pytest.mark.unit
def test_synthetic_tombstones_only_on_final_page() -> None:
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunPagePrepared(
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
            prepared_state_mutation_templates=(_template(),),
            prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
                (_template(),)
            ),
            prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
            prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
            has_more=True,
            delivery_id=_DELIVERY,
            remaining_candidate_remote_ids=("item-c",),
            synthetic_tombstone_remote_ids=("item-z",),
        )


@pytest.mark.unit
def test_candidate_and_tombstone_ids_sorted_unique() -> None:
    collecting = _collecting()
    assert collecting.remaining_candidate_remote_ids == ("item-a", "item-b")
    final_page = _page_prepared(has_more=False)
    assert final_page.synthetic_tombstone_remote_ids == ("item-z",)


@pytest.mark.unit
def test_cursor_hidden_from_repr_and_exceptions() -> None:
    prepared = _page_prepared()
    rendered = repr(prepared)
    assert "https://example.test" not in rendered
    assert "secret" not in rendered
    with pytest.raises(ValidationError) as exc_info:
        KnowledgeReconciliationRunPagePrepared(
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
            prepared_state_mutation_templates=(_template(), _template("item-b")),
            prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
                (_template(), _template("item-b"))
            ),
            prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
            prepared_next_cursor=_CURSOR,
            prepared_next_cursor_fingerprint="d" * 64,
            has_more=True,
            delivery_id=_DELIVERY,
            remaining_candidate_remote_ids=("item-c",),
        )
    assert "https://example.test" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


@pytest.mark.unit
def test_recovery_commands_require_exact_identity() -> None:
    command = KnowledgeReconciliationRecoveryCommand(
        kind=KnowledgeReconciliationRecoveryCommandKind.RESUME_EXACT,
        tenant_id="tenant-1",
        binding_id="binding-1",
        expected_run_id="run-1",
        expected_run_record_version=2,
        expected_phase=KnowledgeReconciliationRunPhase.PAGE_PREPARED,
        operator_reason_code="resume_exact",
    )
    assert command.operator_reason_code == "resume_exact"
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRecoveryCommand(
            kind=KnowledgeReconciliationRecoveryCommandKind.ABORT_PRISTINE,
            tenant_id="tenant-1",
            binding_id="binding-1",
            expected_run_id="run-1",
            expected_run_record_version=1,
            expected_phase=KnowledgeReconciliationRunPhase.COLLECTING,
            operator_reason_code="not a safe code",
        )
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRecoveryCommand(
            kind=KnowledgeReconciliationRecoveryCommandKind.REPAIR_REQUIRED,
            tenant_id="tenant-1",
            binding_id="binding-1",
            expected_run_id="run-1",
            expected_run_record_version=1,
            expected_phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
            operator_reason_code="repair",
            extra="nope",  # type: ignore[call-arg]
        )


@pytest.mark.unit
def test_policy_limits_validated() -> None:
    with pytest.raises(ValidationError):
        KnowledgeReconciliationLimitPolicy(max_reconciliation_candidate_count=0)
    policy = KnowledgeReconciliationLimitPolicy(
        max_reconciliation_candidate_count=1,
        max_reconciliation_candidate_payload_bytes=32,
    )
    with pytest.raises(ValueError):
        validate_reconciliation_candidate_inventory(("a", "b"), policy=policy)
    with pytest.raises(ValueError):
        validate_reconciliation_candidate_inventory(("x" * 40,), policy=policy)
    tiny_policy = KnowledgeReconciliationLimitPolicy(
        max_reconciliation_prepared_intent_payload_bytes=64,
        max_reconciliation_prepared_state_mutation_count=1,
    )
    prepared = _page_prepared()
    with pytest.raises(ValueError):
        validate_reconciliation_prepared_intent(prepared, policy=tiny_policy)


@pytest.mark.unit
def test_sink_and_item_state_receipt_contracts() -> None:
    absent = KnowledgeSyncSinkReceipt(status=KnowledgeSyncSinkReceiptStatus.ABSENT)
    assert absent.delivery_id is None
    applied = KnowledgeSyncSinkReceipt(
        status=KnowledgeSyncSinkReceiptStatus.APPLIED,
        delivery_id=_DELIVERY,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
    )
    assert applied.status is KnowledgeSyncSinkReceiptStatus.APPLIED
    conflict = KnowledgeSyncSinkReceipt(
        status=KnowledgeSyncSinkReceiptStatus.CONFLICT,
        delivery_id=_DELIVERY,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
    )
    assert conflict.status is KnowledgeSyncSinkReceiptStatus.CONFLICT
    unknown = KnowledgeSyncSinkReceipt(
        status=KnowledgeSyncSinkReceiptStatus.UNKNOWN,
        delivery_id=_DELIVERY,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
    )
    assert unknown.status is KnowledgeSyncSinkReceiptStatus.UNKNOWN
    inspector = RecordingSinkReceiptInspector(
        durable={_DELIVERY: _FINGERPRINT},
        unknown_delivery_ids={"unknown"},
    )
    assert isinstance(inspector, KnowledgeSyncSinkReceiptInspector)
    assert (
        inspector.inspect_receipt(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id="missing",
            prepared_batch_payload_fingerprint=_FINGERPRINT,
        ).status
        is KnowledgeSyncSinkReceiptStatus.ABSENT
    )
    item_receipt = KnowledgeRemoteItemStateReceipt(
        status=KnowledgeRemoteItemStateReceiptStatus.COMPLETED,
        delivery_id=_DELIVERY,
        prepared_state_mutations_fingerprint=_FINGERPRINT,
    )
    assert item_receipt.status is KnowledgeRemoteItemStateReceiptStatus.COMPLETED


@pytest.mark.unit
def test_parse_unknown_phase_rejected() -> None:
    payload = _collecting().model_dump(mode="json")
    payload["phase"] = "collecting-extra"
    with pytest.raises(ValueError, match="unknown"):
        parse_knowledge_reconciliation_run(payload)


@pytest.mark.unit
def test_configuration_limit_error_is_not_provider_response() -> None:
    err = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
        safe_message="reconciliation candidate count exceeds configured limit",
        retryable=False,
    )
    assert err.code.value == "configuration_error"
