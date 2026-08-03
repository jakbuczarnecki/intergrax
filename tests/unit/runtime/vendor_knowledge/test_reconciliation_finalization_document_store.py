# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for reconciliation-finalization DocumentStore repositories."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeCursor,
    KnowledgeItemRevision,
)
from intergrax.runtime.vendor_knowledge.sync_contracts import (
    KnowledgeReconciliationRunConflict,
    KnowledgeReconciliationRunRepository,
    KnowledgeSyncCorruptState,
)
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeReconciliationRunRepository,
    DocumentStoreKnowledgeRemoteItemStateRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationLimitPolicy,
    KnowledgeReconciliationMutationSemantic,
    KnowledgeReconciliationPreparedStateMutationTemplate,
    KnowledgeReconciliationRunAborted,
    KnowledgeReconciliationRunCollecting,
    KnowledgeReconciliationRunCompleted,
    KnowledgeReconciliationRunPagePrepared,
    KnowledgeReconciliationRunRecoveryRequired,
    KnowledgeRemoteItemState,
    KnowledgeRemoteItemStateReceiptStatus,
    KnowledgeRemoteItemStatus,
    KnowledgeSyncCheckpoint,
    canonical_prepared_state_mutations_fingerprint,
    knowledge_cursor_fingerprint_sha256,
    reconciliation_run_durable_document_bytes,
    recovery_evidence_from_run,
)

_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
_DELIVERY = "a" * 64
_FINGERPRINT = "b" * 64
_NULL_CURSOR_FP = knowledge_cursor_fingerprint_sha256(None)


def _checkpoint(*, value: str = "cursor-1") -> KnowledgeSyncCheckpoint:
    return KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value=value, version="v1"),
    )


def _collecting(
    *,
    run_id: str = "run-1",
    record_version: int = 1,
    superseded_run_id: str | None = None,
    remote_ids: tuple[str, ...] = ("item-a",),
) -> KnowledgeReconciliationRunCollecting:
    return KnowledgeReconciliationRunCollecting(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id=run_id,
        record_version=record_version,
        created_at=_NOW,
        updated_at=_NOW,
        superseded_run_id=superseded_run_id,
        applied_page_count=0,
        current_input_cursor_fingerprint=_NULL_CURSOR_FP,
        remaining_candidate_remote_ids=remote_ids,
    )


def _template(
    *,
    remote_id: str = "item-1",
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
        remote_id="item-1",
        resulting_status=KnowledgeRemoteItemStatus.ACTIVE,
        revision=KnowledgeItemRevision(version="1"),
        binding_configuration_version=1,
    )


def _page_prepared(
    *, record_version: int = 2
) -> KnowledgeReconciliationRunPagePrepared:
    templates = (_template(), _template(remote_id="item-z", synthetic_tombstone=True))
    return KnowledgeReconciliationRunPagePrepared(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=record_version,
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


def _state(*, delivery_id: str = _DELIVERY) -> KnowledgeRemoteItemState:
    return KnowledgeRemoteItemState(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        remote_id="item-1",
        status=KnowledgeRemoteItemStatus.ACTIVE,
        revision=KnowledgeItemRevision(version="1"),
        last_delivery_id=delivery_id,
    )


@pytest.mark.unit
def test_reconciliation_repository_requires_conditional_document_store() -> None:
    from tests.unit.runtime.vendor_knowledge.test_sync_document_store import (
        _PlainDocumentStore,
    )

    with pytest.raises(TypeError, match="ConditionalDocumentStore"):
        DocumentStoreKnowledgeReconciliationRunRepository(_PlainDocumentStore())


@pytest.mark.unit
def test_initial_create_and_duplicate_rejected() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    run = _collecting()
    repo.create_initial_run(run)
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1") == run
    with pytest.raises(KnowledgeReconciliationRunConflict):
        repo.create_initial_run(_collecting(run_id="run-2"))


@pytest.mark.unit
def test_cas_replace_and_stale_writer_loses() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    initial = _collecting()
    repo.create_initial_run(initial)
    prepared = _page_prepared(record_version=2)
    repo.cas_replace(expected=initial, replacement=prepared)
    loaded = repo.get(tenant_id="tenant-1", binding_id="binding-1")
    assert loaded == prepared
    stale = _collecting(record_version=1)
    with pytest.raises(KnowledgeReconciliationRunConflict):
        repo.cas_replace(expected=stale, replacement=_page_prepared(record_version=2))


@pytest.mark.unit
def test_record_version_must_increment_exactly_once() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    initial = _collecting()
    repo.create_initial_run(initial)
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.cas_replace(
            expected=initial,
            replacement=_page_prepared(record_version=3),
        )


@pytest.mark.unit
def test_immutable_identity_cannot_change() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    initial = _collecting()
    repo.create_initial_run(initial)
    mutated = initial.model_copy(update={"provider_id": "other", "record_version": 2})
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.cas_replace(expected=initial, replacement=mutated)


@pytest.mark.unit
def test_invalid_phase_transition_rejected() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    initial = _collecting()
    repo.create_initial_run(initial)
    completed = KnowledgeReconciliationRunCompleted(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=2,
        created_at=_NOW,
        updated_at=_NOW,
        committed_completed_checkpoint=_checkpoint(),
        final_delivery_id=_DELIVERY,
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.cas_replace(expected=initial, replacement=completed)


@pytest.mark.unit
def test_terminal_supersession_completed_and_aborted() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    completed = KnowledgeReconciliationRunCompleted(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-old",
        record_version=1,
        created_at=_NOW,
        updated_at=_NOW,
        committed_completed_checkpoint=_checkpoint(),
        final_delivery_id=_DELIVERY,
    )
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.reconciliation_run.v1:tenant-1",
            row_key="binding:binding-1",
            data={
                "schema_version": "vendor_knowledge.reconciliation_run.v1",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "record_version": completed.record_version,
                "run": completed.model_dump(mode="json"),
            },
        )
    )
    replacement = _collecting(run_id="run-new", superseded_run_id="run-old")
    repo.cas_supersede_terminal(expected=completed, replacement=replacement)
    loaded = repo.get(tenant_id="tenant-1", binding_id="binding-1")
    assert loaded == replacement
    aborted = KnowledgeReconciliationRunAborted(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-aborted",
        record_version=1,
        created_at=_NOW,
        updated_at=_NOW,
        operator_reason_code="operator_abort",
    )
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.reconciliation_run.v1:tenant-1",
            row_key="binding:binding-1",
            data={
                "schema_version": "vendor_knowledge.reconciliation_run.v1",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "record_version": aborted.record_version,
                "run": aborted.model_dump(mode="json"),
            },
        )
    )
    repo.cas_supersede_terminal(
        expected=aborted,
        replacement=_collecting(
            run_id="run-after-abort", superseded_run_id="run-aborted"
        ),
    )


@pytest.mark.unit
def test_recovery_required_supersession_forbidden() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    recovery = KnowledgeReconciliationRunRecoveryRequired(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-recovery",
        record_version=1,
        created_at=_NOW,
        updated_at=_NOW,
        recovery_reason_code="provider_page_mismatch",
        recovery_evidence=recovery_evidence_from_run(
            _collecting(remote_ids=("item-a",))
        ),
    )
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.reconciliation_run.v1:tenant-1",
            row_key="binding:binding-1",
            data={
                "schema_version": "vendor_knowledge.reconciliation_run.v1",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "record_version": recovery.record_version,
                "run": recovery.model_dump(mode="json"),
            },
        )
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.cas_supersede_terminal(
            expected=recovery,
            replacement=_collecting(run_id="run-new", superseded_run_id="run-recovery"),
        )


@pytest.mark.unit
def test_cross_tenant_and_binding_corruption_rejected() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    run = _collecting()
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.reconciliation_run.v1:tenant-1",
            row_key="binding:binding-1",
            data={
                "schema_version": "vendor_knowledge.reconciliation_run.v1",
                "tenant_id": "other-tenant",
                "binding_id": "binding-1",
                "record_version": 1,
                "run": run.model_dump(mode="json"),
            },
        )
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.get(tenant_id="tenant-1", binding_id="binding-1")
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.reconciliation_run.v1:tenant-1",
            row_key="binding:other-binding",
            data={
                "schema_version": "vendor_knowledge.reconciliation_run.v1",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "record_version": 1,
                "run": run.model_dump(mode="json"),
            },
        )
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.get(tenant_id="tenant-1", binding_id="binding-1")


@pytest.mark.unit
def test_malformed_and_unknown_schema_rejected() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    store.put(
        DocumentRecord(
            partition_key="vendor_knowledge.reconciliation_run.v1:tenant-1",
            row_key="binding:binding-1",
            data={
                "schema_version": "vendor_knowledge.reconciliation_run.v9",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "record_version": 1,
                "run": {"phase": "collecting"},
            },
        )
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.get(tenant_id="tenant-1", binding_id="binding-1")


@pytest.mark.unit
def test_no_ttl_on_reconciliation_run_document() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    repo.create_initial_run(_collecting())
    document = store.get(
        "vendor_knowledge.reconciliation_run.v1:tenant-1",
        "binding:binding-1",
    )
    assert document is not None
    assert document.ttl_seconds is None


@pytest.mark.unit
def test_policy_overflow_leaves_prior_run_unchanged() -> None:
    store = InMemoryDocumentStore()
    policy = KnowledgeReconciliationLimitPolicy(max_reconciliation_candidate_count=1)
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store, policy=policy)
    initial = _collecting(remote_ids=("item-a",))
    repo.create_initial_run(initial)
    overflow = _collecting(remote_ids=("item-a", "item-b"))
    overflow = overflow.model_copy(update={"record_version": 2})
    with pytest.raises(VendorKnowledgeError) as exc_info:
        repo.cas_replace(expected=initial, replacement=overflow)
    assert exc_info.value.code.value == "configuration_error"
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1") == initial


@pytest.mark.unit
def test_prepared_intent_overflow_rejected_without_write() -> None:
    store = InMemoryDocumentStore()
    policy = KnowledgeReconciliationLimitPolicy(
        max_reconciliation_prepared_intent_payload_bytes=128,
        max_reconciliation_prepared_state_mutation_count=1,
    )
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store, policy=policy)
    initial = _collecting()
    repo.create_initial_run(initial)
    with pytest.raises(VendorKnowledgeError):
        repo.cas_replace(expected=initial, replacement=_page_prepared(record_version=2))


@pytest.mark.unit
def test_item_state_receipt_inspection_states() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    mutations_fp = canonical_prepared_state_mutations_fingerprint((_template(),))
    absent = repo.inspect_delivery_receipt(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    assert absent.status is KnowledgeRemoteItemStateReceiptStatus.ABSENT
    partition = "vendor_knowledge.remote_item.v1:tenant-1:binding-1"
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=f"delivery:{_DELIVERY}",
            data={
                "schema_version": "vendor_knowledge.delivery_marker.v2",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "delivery_id": _DELIVERY,
                "batch_fingerprint": _FINGERPRINT,
                "prepared_state_mutations_fingerprint": mutations_fp,
                "status": "applying",
                "record_version": "rv-1",
            },
        )
    )
    applying = repo.inspect_delivery_receipt(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    assert applying.status is KnowledgeRemoteItemStateReceiptStatus.APPLYING
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=f"delivery:{_DELIVERY}",
            data={
                "schema_version": "vendor_knowledge.delivery_marker.v2",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "delivery_id": _DELIVERY,
                "batch_fingerprint": _FINGERPRINT,
                "prepared_state_mutations_fingerprint": mutations_fp,
                "status": "completed",
                "record_version": "rv-2",
            },
        )
    )
    completed = repo.inspect_delivery_receipt(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    assert completed.status is KnowledgeRemoteItemStateReceiptStatus.COMPLETED
    conflict = repo.inspect_delivery_receipt(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        prepared_state_mutations_fingerprint="c" * 64,
    )
    assert conflict.status is KnowledgeRemoteItemStateReceiptStatus.CONFLICT


@pytest.mark.unit
def test_receipt_inspection_does_not_mutate_state() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    mutations_fp = canonical_prepared_state_mutations_fingerprint((_template(),))
    before = store.query("vendor_knowledge.remote_item.v1:tenant-1:binding-1")
    repo.inspect_delivery_receipt(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    after = store.query("vendor_knowledge.remote_item.v1:tenant-1:binding-1")
    assert after.total == before.total


@pytest.mark.unit
def test_malformed_receipt_marker_fails_closed() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    partition = "vendor_knowledge.remote_item.v1:tenant-1:binding-1"
    store.put(
        DocumentRecord(
            partition_key=partition,
            row_key=f"delivery:{_DELIVERY}",
            data={
                "schema_version": "vendor_knowledge.delivery_marker.v2",
                "tenant_id": "tenant-1",
                "binding_id": "binding-1",
                "delivery_id": _DELIVERY,
                "batch_fingerprint": _FINGERPRINT,
                "status": "completed",
                "record_version": "rv-1",
            },
        )
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.inspect_delivery_receipt(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=_DELIVERY,
            prepared_state_mutations_fingerprint=_FINGERPRINT,
        )


@pytest.mark.unit
def test_existing_apply_batch_replay_unchanged() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    states = (_state(),)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        states=states,
    )
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        states=states,
    )
    assert (
        repo.get(tenant_id="tenant-1", binding_id="binding-1", remote_id="item-1")
        == states[0]
    )


@pytest.mark.unit
def test_repository_protocol_runtime_checkable() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    assert isinstance(repo, KnowledgeReconciliationRunRepository)


@pytest.mark.unit
def test_base_checkpoint_immutable_across_cas() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    base = _checkpoint()
    initial = _collecting().model_copy(
        update={"expected_base_completed_checkpoint": base}
    )
    repo.create_initial_run(initial)
    prepared = _page_prepared(record_version=2).model_copy(
        update={"expected_base_completed_checkpoint": base}
    )
    repo.cas_replace(expected=initial, replacement=prepared)
    mutated = prepared.model_copy(
        update={
            "record_version": 3,
            "expected_base_completed_checkpoint": _checkpoint(value="mutated"),
        }
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.cas_replace(expected=prepared, replacement=mutated)


@pytest.mark.unit
def test_cas_monotonicity_page_prepared_to_collecting() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    initial = _collecting()
    repo.create_initial_run(initial)
    prepared = _page_prepared(record_version=2)
    repo.cas_replace(expected=initial, replacement=prepared)
    bad_collecting = _collecting(
        record_version=3,
        remote_ids=(),
    ).model_copy(
        update={
            "applied_page_count": 0,
            "last_applied_delivery_id": None,
            "current_input_cursor_fingerprint": _NULL_CURSOR_FP,
        }
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.cas_replace(expected=prepared, replacement=bad_collecting)


@pytest.mark.unit
def test_recovery_transition_retains_exact_evidence() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store)
    initial = _collecting()
    repo.create_initial_run(initial)
    prepared = _page_prepared(record_version=2)
    repo.cas_replace(expected=initial, replacement=prepared)
    recovery = KnowledgeReconciliationRunRecoveryRequired(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-1",
        record_version=3,
        created_at=_NOW,
        updated_at=_NOW,
        applied_page_count=0,
        recovery_reason_code="provider_page_mismatch",
        recovery_evidence=recovery_evidence_from_run(prepared),
    )
    repo.cas_replace(expected=prepared, replacement=recovery)
    loaded = repo.get(tenant_id="tenant-1", binding_id="binding-1")
    assert isinstance(loaded, KnowledgeReconciliationRunRecoveryRequired)
    assert loaded.recovery_evidence.delivery_id == _DELIVERY


@pytest.mark.unit
def test_real_apply_inspect_receipt_completed() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    templates = (_template(),)
    mutations_fp = canonical_prepared_state_mutations_fingerprint(templates)
    states = (_state(),)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        states=states,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    receipt = repo.inspect_delivery_receipt(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    assert receipt.status is KnowledgeRemoteItemStateReceiptStatus.COMPLETED


@pytest.mark.unit
def test_v2_apply_idempotent_and_conflict() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    mutations_fp = canonical_prepared_state_mutations_fingerprint((_template(),))
    states = (_state(),)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        states=states,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        states=states,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=_DELIVERY,
            states=states,
            prepared_state_mutations_fingerprint="c" * 64,
        )


@pytest.mark.unit
def test_legacy_v1_marker_cannot_prove_v2_receipt() -> None:
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeRemoteItemStateRepository(store)
    states = (_state(),)
    repo.apply_batch(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        states=states,
    )
    mutations_fp = canonical_prepared_state_mutations_fingerprint((_template(),))
    receipt = repo.inspect_delivery_receipt(
        tenant_id="tenant-1",
        binding_id="binding-1",
        delivery_id=_DELIVERY,
        prepared_state_mutations_fingerprint=mutations_fp,
    )
    assert receipt.status is KnowledgeRemoteItemStateReceiptStatus.CONFLICT
    with pytest.raises(KnowledgeSyncCorruptState):
        repo.apply_batch(
            tenant_id="tenant-1",
            binding_id="binding-1",
            delivery_id=_DELIVERY,
            states=states,
            prepared_state_mutations_fingerprint=mutations_fp,
        )


@pytest.mark.unit
def test_remote_id_overflow_leaves_prior_run_unchanged() -> None:
    store = InMemoryDocumentStore()
    policy = KnowledgeReconciliationLimitPolicy(max_reconciliation_remote_id_bytes=4)
    repo = DocumentStoreKnowledgeReconciliationRunRepository(store, policy=policy)
    initial = _collecting(remote_ids=("abcd",))
    repo.create_initial_run(initial)
    overflow = _collecting(remote_ids=("abcde",)).model_copy(
        update={"record_version": 2}
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        repo.cas_replace(expected=initial, replacement=overflow)
    assert exc_info.value.code.value == "configuration_error"
    assert repo.get(tenant_id="tenant-1", binding_id="binding-1") == initial


@pytest.mark.unit
def test_prepared_wrapper_limit_rejects_run_only_fit() -> None:
    prepared = _page_prepared(record_version=2)
    wrapper_len = len(reconciliation_run_durable_document_bytes(prepared))
    run_only_len = len(prepared.model_dump_json().encode("utf-8"))
    assert wrapper_len > run_only_len
    store = InMemoryDocumentStore()
    repo = DocumentStoreKnowledgeReconciliationRunRepository(
        store,
        policy=KnowledgeReconciliationLimitPolicy(
            max_reconciliation_prepared_intent_payload_bytes=run_only_len
        ),
    )
    initial = _collecting()
    repo.create_initial_run(initial)
    with pytest.raises(VendorKnowledgeError):
        repo.cas_replace(expected=initial, replacement=prepared)
