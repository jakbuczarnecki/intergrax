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
    KnowledgeReconciliationMutationSemantic,
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
    reconciliation_run_durable_document_bytes,
    recovery_evidence_from_run,
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


def _checkpoint(
    *,
    cursor_value: str = "cursor-1",
    config_version: int = 1,
) -> KnowledgeSyncCheckpoint:
    return KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=config_version,
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
    *,
    synthetic_tombstone: bool = False,
) -> KnowledgeReconciliationPreparedStateMutationTemplate:
    if synthetic_tombstone:
        return KnowledgeReconciliationPreparedStateMutationTemplate(
            remote_id=remote_id,
            resulting_status=KnowledgeRemoteItemStatus.DELETED,
            binding_configuration_version=1,
            reconciliation_semantic=KnowledgeReconciliationMutationSemantic.ABSENT_FROM_COMPLETED_SYNCHRONIZED_SOURCE_INVENTORY,
        )
    return KnowledgeReconciliationPreparedStateMutationTemplate(
        remote_id=remote_id,
        resulting_status=KnowledgeRemoteItemStatus.ACTIVE,
        revision=KnowledgeItemRevision(version="1"),
        binding_configuration_version=1,
    )


def _page_prepared(*, has_more: bool = True) -> KnowledgeReconciliationRunPagePrepared:
    templates: tuple[KnowledgeReconciliationPreparedStateMutationTemplate, ...]
    if has_more:
        templates = (_template("item-a"), _template("item-b"))
        synthetic_ids: tuple[str, ...] = ()
    else:
        templates = (
            _template("item-a"),
            _template("item-b"),
            _template("item-z", synthetic_tombstone=True),
        )
        synthetic_ids = ("item-z",)
    mutations_fp = canonical_prepared_state_mutations_fingerprint(templates)
    next_cursor = _CURSOR if has_more else None
    next_cursor_fp = _CURSOR_FP if has_more else _NULL_CURSOR_FP
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
        prepared_next_cursor_fingerprint=next_cursor_fp,
        prepared_next_cursor=next_cursor,
        prepared_page_size=100,
        has_more=has_more,
        delivery_id=_DELIVERY,
        remaining_candidate_remote_ids=("item-c",) if has_more else ("item-z",),
        synthetic_tombstone_remote_ids=synthetic_ids,
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
        expected_base_completed_checkpoint=checkpoint,
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
        expected_base_completed_checkpoint=checkpoint,
        recovery_reason_code="provider_page_mismatch",
        recovery_evidence=recovery_evidence_from_run(finalizing),
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


@pytest.mark.unit
def test_base_checkpoint_none_for_first_sync() -> None:
    collecting = _collecting()
    assert collecting.expected_base_completed_checkpoint is None
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
        intended_final_completed_checkpoint=_checkpoint(),
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            _checkpoint()
        ),
        expected_previous_completed_checkpoint=None,
        final_delivery_id=_DELIVERY,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
    )
    assert finalizing.expected_previous_completed_checkpoint is None


@pytest.mark.unit
def test_base_checkpoint_tenant_binding_mismatch_rejected() -> None:
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
            expected_base_completed_checkpoint=KnowledgeSyncCheckpoint(
                tenant_id="other",
                binding_id="binding-1",
                binding_configuration_version=1,
                cursor=KnowledgeCursor(value="c", version="v1"),
            ),
        )


@pytest.mark.unit
def test_finalizing_with_existing_base_checkpoint() -> None:
    base = _checkpoint(cursor_value="base")
    finalizing = KnowledgeReconciliationRunFinalizing(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=2,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=3,
        created_at=_NOW,
        updated_at=_NOW,
        applied_page_count=1,
        last_applied_delivery_id=_DELIVERY,
        expected_base_completed_checkpoint=base,
        intended_final_completed_checkpoint=_checkpoint(
            cursor_value="final",
            config_version=2,
        ),
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            _checkpoint(cursor_value="final", config_version=2)
        ),
        expected_previous_completed_checkpoint=base,
        final_delivery_id=_DELIVERY,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
    )
    assert finalizing.expected_base_completed_checkpoint == base


@pytest.mark.unit
def test_page_prepared_recovery_evidence_round_trip() -> None:
    prepared = _page_prepared(has_more=False)
    evidence = recovery_evidence_from_run(prepared)
    run = KnowledgeReconciliationRunRecoveryRequired(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=3,
        created_at=_NOW,
        updated_at=_NOW,
        recovery_reason_code="provider_page_mismatch",
        recovery_evidence=evidence,
    )
    restored = parse_knowledge_reconciliation_run(run.model_dump(mode="json"))
    assert isinstance(restored, KnowledgeReconciliationRunRecoveryRequired)
    assert restored.recovery_evidence == evidence
    assert isinstance(
        restored.recovery_evidence,
        type(evidence),
    )


@pytest.mark.unit
def test_finalizing_recovery_evidence_round_trip() -> None:
    base = _checkpoint()
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
        expected_base_completed_checkpoint=base,
        intended_final_completed_checkpoint=base,
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            base
        ),
        expected_previous_completed_checkpoint=base,
        final_delivery_id=_DELIVERY,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
    )
    evidence = recovery_evidence_from_run(finalizing)
    restored = parse_knowledge_reconciliation_run(
        KnowledgeReconciliationRunRecoveryRequired(
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
            expected_base_completed_checkpoint=base,
            recovery_reason_code="checkpoint_mismatch",
            recovery_evidence=evidence,
        ).model_dump(mode="json")
    )
    assert isinstance(restored, KnowledgeReconciliationRunRecoveryRequired)
    assert restored.recovery_evidence.final_delivery_id == _DELIVERY


@pytest.mark.unit
def test_recovery_repr_and_exceptions_remain_secret_free() -> None:
    prepared = _page_prepared()
    evidence = recovery_evidence_from_run(prepared)
    rendered = repr(evidence)
    assert "https://example.test" not in rendered
    assert "secret" not in rendered
    with pytest.raises(ValidationError) as exc_info:
        KnowledgeReconciliationRunRecoveryRequired(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            provider_id="example",
            source_kind="issues",
            run_id="run-1",
            record_version=3,
            created_at=_NOW,
            updated_at=_NOW,
            recovery_reason_code="provider_page_mismatch",
            recovery_evidence=evidence.model_copy(
                update={"prepared_next_cursor_fingerprint": "d" * 64}
            ),
        )
    assert "https://example.test" not in str(exc_info.value)


@pytest.mark.unit
def test_malformed_recovery_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunRecoveryRequired(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            provider_id="example",
            source_kind="issues",
            run_id="run-1",
            record_version=3,
            created_at=_NOW,
            updated_at=_NOW,
            recovery_reason_code="provider_page_mismatch",
            recovery_evidence={
                "origin_phase": "page_prepared",
                "delivery_id": "short",
            },
        )


@pytest.mark.unit
def test_remote_id_utf8_byte_limit() -> None:
    policy = KnowledgeReconciliationLimitPolicy(max_reconciliation_remote_id_bytes=4)
    exact = "é" * 2
    assert len(exact.encode("utf-8")) == 4
    validate_reconciliation_candidate_inventory((exact,), policy=policy)
    with pytest.raises(ValueError):
        validate_reconciliation_candidate_inventory((exact + "x",), policy=policy)


@pytest.mark.unit
def test_mutation_templates_require_unique_sorted_remote_ids() -> None:
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
            prepared_state_mutation_templates=(
                _template("item-b"),
                _template("item-a"),
            ),
            prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
                (_template("item-b"), _template("item-a"))
            ),
            prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
            prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
            has_more=True,
            delivery_id=_DELIVERY,
            remaining_candidate_remote_ids=("item-c",),
        )
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
            prepared_state_mutation_templates=(
                _template("item-a"),
                _template("item-a"),
            ),
            prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
                (_template("item-a"), _template("item-a"))
            ),
            prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
            prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
            has_more=True,
            delivery_id=_DELIVERY,
            remaining_candidate_remote_ids=("item-c",),
        )


@pytest.mark.unit
def test_synthetic_tombstone_semantic_marker_required() -> None:
    templates = (
        _template("item-a"),
        _template("item-b"),
        _template("item-z"),
    )
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
            prepared_state_mutation_templates=templates,
            prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
                templates
            ),
            prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
            prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
            has_more=False,
            delivery_id=_DELIVERY,
            remaining_candidate_remote_ids=(),
            synthetic_tombstone_remote_ids=("item-z",),
        )


@pytest.mark.unit
def test_prepared_intent_measures_complete_durable_wrapper() -> None:
    prepared = _page_prepared()
    wrapper = reconciliation_run_durable_document_bytes(prepared)
    run_only = len(prepared.model_dump_json().encode("utf-8"))
    assert len(wrapper) >= run_only
    policy = KnowledgeReconciliationLimitPolicy(
        max_reconciliation_prepared_intent_payload_bytes=len(wrapper)
    )
    validate_reconciliation_prepared_intent(prepared, policy=policy)
    with pytest.raises(ValueError):
        validate_reconciliation_prepared_intent(
            prepared,
            policy=KnowledgeReconciliationLimitPolicy(
                max_reconciliation_prepared_intent_payload_bytes=len(wrapper) - 1
            ),
        )


@pytest.mark.unit
def test_final_page_requires_exact_tombstone_equality() -> None:
    templates = (
        _template("item-a"),
        _template("item-z", synthetic_tombstone=True),
    )
    mutations_fp = canonical_prepared_state_mutations_fingerprint(templates)
    base = dict(
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
        prepared_state_mutations_fingerprint=mutations_fp,
        prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
        prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
        prepared_page_size=100,
        has_more=False,
        delivery_id=_DELIVERY,
    )
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunPagePrepared(
            **base,
            remaining_candidate_remote_ids=("item-z",),
            synthetic_tombstone_remote_ids=(),
        )
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunPagePrepared(
            **base,
            remaining_candidate_remote_ids=("item-z", "item-y"),
            synthetic_tombstone_remote_ids=("item-z",),
        )
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunPagePrepared(
            **base,
            remaining_candidate_remote_ids=("item-z",),
            synthetic_tombstone_remote_ids=("item-z", "item-y"),
        )
    empty_templates = (_template("item-a"),)
    empty_mutations_fp = canonical_prepared_state_mutations_fingerprint(empty_templates)
    empty_base = {
        **base,
        "prepared_state_mutation_templates": empty_templates,
        "prepared_state_mutations_fingerprint": empty_mutations_fp,
    }
    accepted = KnowledgeReconciliationRunPagePrepared(
        **empty_base,
        remaining_candidate_remote_ids=(),
        synthetic_tombstone_remote_ids=(),
    )
    assert accepted.synthetic_tombstone_remote_ids == ()
    accepted_final = KnowledgeReconciliationRunPagePrepared(
        **base,
        remaining_candidate_remote_ids=("item-z",),
        synthetic_tombstone_remote_ids=("item-z",),
    )
    assert accepted_final.remaining_candidate_remote_ids == ("item-z",)


@pytest.mark.unit
def test_custom_remote_id_policy_above_default() -> None:
    large_id = "x" * 3000
    assert len(large_id.encode("utf-8")) > 2048
    policy = KnowledgeReconciliationLimitPolicy(max_reconciliation_remote_id_bytes=4096)
    validate_reconciliation_candidate_inventory((large_id,), policy=policy)
    default_policy = KnowledgeReconciliationLimitPolicy()
    with pytest.raises(ValueError):
        validate_reconciliation_candidate_inventory((large_id,), policy=default_policy)
    exact_policy = KnowledgeReconciliationLimitPolicy(
        max_reconciliation_remote_id_bytes=2048
    )
    exact_id = "y" * 2048
    validate_reconciliation_candidate_inventory((exact_id,), policy=exact_policy)
    below_policy = KnowledgeReconciliationLimitPolicy(
        max_reconciliation_remote_id_bytes=8
    )
    with pytest.raises(ValueError):
        validate_reconciliation_candidate_inventory(("123456789",), policy=below_policy)


@pytest.mark.unit
def test_recovery_evidence_self_validation_rejects_mismatched_base() -> None:
    prepared = _page_prepared(has_more=False)
    evidence = recovery_evidence_from_run(prepared)
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunRecoveryRequired(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            provider_id="example",
            source_kind="issues",
            run_id="run-1",
            record_version=3,
            created_at=_NOW,
            updated_at=_NOW,
            recovery_reason_code="provider_page_mismatch",
            recovery_evidence=evidence,
            expected_base_completed_checkpoint=_checkpoint(),
        )


@pytest.mark.unit
def test_final_page_empty_templates_with_candidates_rejected() -> None:
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
            prepared_state_mutation_templates=(),
            prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
                ()
            ),
            prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
            prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
            has_more=False,
            delivery_id=_DELIVERY,
            remaining_candidate_remote_ids=("item-z",),
            synthetic_tombstone_remote_ids=("item-z",),
        )


@pytest.mark.unit
def test_final_page_missing_synthetic_template_rejected() -> None:
    templates = (_template("item-a"),)
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
            prepared_state_mutation_templates=templates,
            prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
                templates
            ),
            prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
            prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
            has_more=False,
            delivery_id=_DELIVERY,
            remaining_candidate_remote_ids=("item-z",),
            synthetic_tombstone_remote_ids=("item-z",),
        )


@pytest.mark.unit
def test_final_page_empty_candidates_and_templates_accepted() -> None:
    accepted = KnowledgeReconciliationRunPagePrepared(
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
        prepared_state_mutation_templates=(),
        prepared_state_mutations_fingerprint=canonical_prepared_state_mutations_fingerprint(
            ()
        ),
        prepared_proposed_checkpoint_fingerprint=_NULL_CURSOR_FP,
        prepared_next_cursor_fingerprint=_NULL_CURSOR_FP,
        prepared_page_size=100,
        has_more=False,
        delivery_id=_DELIVERY,
        remaining_candidate_remote_ids=(),
        synthetic_tombstone_remote_ids=(),
    )
    assert accepted.prepared_state_mutation_templates == ()


@pytest.mark.unit
def test_recovery_evidence_rejects_synthetic_ids_without_templates() -> None:
    prepared = _page_prepared(has_more=False)
    evidence = recovery_evidence_from_run(prepared)
    corrupt = evidence.model_copy(
        update={
            "prepared_state_mutation_templates": (),
            "prepared_state_mutations_fingerprint": canonical_prepared_state_mutations_fingerprint(
                ()
            ),
        }
    )
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunRecoveryRequired(
            tenant_id="tenant-1",
            binding_id="binding-1",
            binding_configuration_version=1,
            provider_id="example",
            source_kind="issues",
            run_id="run-1",
            record_version=3,
            created_at=_NOW,
            updated_at=_NOW,
            recovery_reason_code="provider_page_mismatch",
            recovery_evidence=corrupt,
        )


@pytest.mark.unit
def test_finalizing_rejects_intended_checkpoint_config_version_mismatch() -> None:
    checkpoint = _checkpoint(config_version=2)
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunFinalizing(
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
            expected_previous_completed_checkpoint=None,
            final_delivery_id=_DELIVERY,
            prepared_batch_payload_fingerprint=_FINGERPRINT,
        )


@pytest.mark.unit
def test_completed_rejects_committed_checkpoint_config_version_mismatch() -> None:
    checkpoint = _checkpoint(config_version=2)
    with pytest.raises(ValidationError):
        KnowledgeReconciliationRunCompleted(
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


@pytest.mark.unit
def test_finalizing_allows_older_previous_checkpoint_config_version() -> None:
    base = _checkpoint(cursor_value="base", config_version=1)
    intended = _checkpoint(cursor_value="final", config_version=2)
    finalizing = KnowledgeReconciliationRunFinalizing(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=2,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=3,
        created_at=_NOW,
        updated_at=_NOW,
        applied_page_count=1,
        last_applied_delivery_id=_DELIVERY,
        expected_base_completed_checkpoint=base,
        intended_final_completed_checkpoint=intended,
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            intended
        ),
        expected_previous_completed_checkpoint=base,
        final_delivery_id=_DELIVERY,
        prepared_batch_payload_fingerprint=_FINGERPRINT,
    )
    assert finalizing.expected_previous_completed_checkpoint == base
    assert finalizing.intended_final_completed_checkpoint == intended
