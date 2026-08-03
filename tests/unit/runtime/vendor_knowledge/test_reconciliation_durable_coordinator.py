# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable reconciliation coordinator, failure-window and fingerprint tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChangeKind,
    KnowledgeCursor,
)
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationRunPhase,
    KnowledgeSyncMode,
    reconciliation_delivery_id,
    reconciliation_prepared_batch_payload_fingerprint,
    reconciliation_provider_page_fingerprint,
)
from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
    derive_reconciliation_run_id,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    InMemoryCandidateInventoryRepository,
    InMemoryCheckpointRepository,
    InMemoryLeaseRepository,
    InMemoryReconciliationRunRepository,
    InMemoryRemoteItemStateRepository,
    RecordingBindingService,
    RecordingFacade,
    RecordingSinkReceiptInspector,
    make_binding,
    make_change,
    make_page,
)


def _durable_coordinator(
    *,
    facade: RecordingFacade | None = None,
    state: InMemoryRemoteItemStateRepository | None = None,
    checkpoint: InMemoryCheckpointRepository | None = None,
    runs: InMemoryReconciliationRunRepository | None = None,
    inventory_incomplete: bool = False,
) -> tuple[
    VendorKnowledgeSyncCoordinator,
    RecordingFacade,
    InMemoryRemoteItemStateRepository,
    InMemoryCheckpointRepository,
    InMemoryReconciliationRunRepository,
    IdempotentRecordingSink,
    RecordingSinkReceiptInspector,
]:
    binding = make_binding()
    facade = facade or RecordingFacade()
    state = state or InMemoryRemoteItemStateRepository()
    checkpoint = checkpoint or InMemoryCheckpointRepository()
    runs = runs or InMemoryReconciliationRunRepository()
    sink = IdempotentRecordingSink()
    inspector = RecordingSinkReceiptInspector()
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=InMemoryLeaseRepository(),
        checkpoint_repository=checkpoint,
        item_state_repository=state,
        sink=sink,
        lease_ttl_seconds=30,
        reconciliation_run_repository=runs,
        candidate_inventory_repository=InMemoryCandidateInventoryRepository(
            state_repository=state,
            force_incomplete=inventory_incomplete,
        ),
        sink_receipt_inspector=inspector,
    )
    return coordinator, facade, state, checkpoint, runs, sink, inspector


@pytest.mark.unit
@pytest.mark.asyncio
async def test_durable_reconciliation_two_page_flow() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    final_cp = KnowledgeCursor(value="reconcile-final")
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            reconciliation=True,
            full_inventory=False,
            incremental_changes=False,
            content_fetch=True,
        ),
        pages_by_cursor={
            None: make_page(
                changes=(make_change(remote_id="item-1"),),
                has_more=True,
                next_cursor=reconcile_cp1,
                proposed_checkpoint=reconcile_cp1,
            ),
            "reconcile-cp-1": make_page(
                changes=(make_change(remote_id="item-2"),),
                proposed_checkpoint=final_cp,
                has_more=False,
            ),
        },
    )
    coordinator, facade_rec, _, checkpoint, runs, sink, _ = _durable_coordinator(
        facade=facade
    )
    operation_id = "op-recon-1"
    first = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=True,
        operation_id=operation_id,
        page_size=50,
    )
    assert first.has_more is True
    assert first.checkpoint_advanced is False
    assert first.delivery_id is not None
    assert checkpoint.commit_calls == []
    run = runs.runs[("tenant-1", "binding-1")]
    assert run.phase is KnowledgeReconciliationRunPhase.COLLECTING
    second = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=first.delivery_id,
        page_size=50,
    )
    assert second.has_more is False
    assert second.checkpoint_advanced is True
    assert len(facade_rec.read_calls) == 2
    assert len(sink.calls) == 2
    assert len(checkpoint.commit_calls) == 1
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor == final_cp


@pytest.mark.unit
@pytest.mark.asyncio
async def test_incremental_blocked_while_reconciliation_active() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            reconciliation=True, content_fetch=True
        ),
        pages_by_cursor={
            None: make_page(
                changes=(make_change(remote_id="item-1"),),
                has_more=True,
                next_cursor=reconcile_cp1,
                proposed_checkpoint=reconcile_cp1,
            ),
        },
    )
    coordinator, *_ = _durable_coordinator(facade=facade)
    await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=True,
        operation_id="op-block",
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert "blocked" in exc_info.value.safe_message.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_legacy_incomplete_inventory_rejected() -> None:
    state = InMemoryRemoteItemStateRepository()
    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeRemoteItemState,
        KnowledgeRemoteItemStatus,
    )
    from intergrax.runtime.vendor_knowledge.models import KnowledgeItemRevision

    state.states[("tenant-1", "binding-1", "legacy-item")] = KnowledgeRemoteItemState(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        remote_id="legacy-item",
        status=KnowledgeRemoteItemStatus.ACTIVE,
        revision=KnowledgeItemRevision(version="1"),
        last_delivery_id="a" * 64,
    )
    coordinator, facade, *_ = _durable_coordinator(
        state=state,
        inventory_incomplete=True,
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-legacy",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_continuation_rejected_without_provider_read() -> None:
    coordinator, facade, *_ = _durable_coordinator()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=False,
            operation_id="op-stale",
            trigger_delivery_id="b" * 64,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_operation_retry_resumes_collecting_run() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            reconciliation=True, content_fetch=True
        ),
        pages_by_cursor={
            None: make_page(
                changes=(make_change(remote_id="item-1"),),
                has_more=True,
                next_cursor=reconcile_cp1,
                proposed_checkpoint=reconcile_cp1,
            ),
        },
    )
    coordinator, _, _, _, runs, *_ = _durable_coordinator(facade=facade)
    operation_id = "op-retry"
    await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=True,
        operation_id=operation_id,
    )
    run_after_first = runs.runs[("tenant-1", "binding-1")]
    await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=True,
        operation_id=operation_id,
    )
    assert runs.runs[("tenant-1", "binding-1")].run_id == run_after_first.run_id


@pytest.mark.unit
def test_fingerprints_and_delivery_are_deterministic() -> None:
    from intergrax.runtime.vendor_knowledge.bindings import to_source_ref

    binding = make_binding()
    source = to_source_ref(binding)
    input_fp = "0" * 64
    proposed_fp = "1" * 64
    next_fp = "2" * 64
    provider_fp = reconciliation_provider_page_fingerprint(
        input_cursor_fingerprint=input_fp,
        has_more=False,
        proposed_checkpoint_fingerprint=proposed_fp,
        next_cursor_fingerprint=next_fp,
        changes=(("item-1", KnowledgeChangeKind.UPSERT, None),),
    )
    batch_fp = reconciliation_prepared_batch_payload_fingerprint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        mode=KnowledgeSyncMode.RECONCILIATION,
        run_id="run-1",
        source=source,
        has_more=False,
        envelopes=(),
        prepared_state_mutations_fingerprint="3" * 64,
        provider_page_fingerprint=provider_fp,
        input_cursor_fingerprint=input_fp,
        proposed_checkpoint_fingerprint=proposed_fp,
        next_cursor_fingerprint=next_fp,
    )
    delivery = reconciliation_delivery_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        mode=KnowledgeSyncMode.RECONCILIATION,
        run_id="run-1",
        provider_page_fingerprint=provider_fp,
        prepared_batch_payload_fingerprint=batch_fp,
        prepared_state_mutations_fingerprint="3" * 64,
        input_cursor_fingerprint=input_fp,
        proposed_checkpoint_fingerprint=proposed_fp,
        next_cursor_fingerprint=next_fp,
    )
    delivery_again = reconciliation_delivery_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        mode=KnowledgeSyncMode.RECONCILIATION,
        run_id="run-1",
        provider_page_fingerprint=provider_fp,
        prepared_batch_payload_fingerprint=batch_fp,
        prepared_state_mutations_fingerprint="3" * 64,
        input_cursor_fingerprint=input_fp,
        proposed_checkpoint_fingerprint=proposed_fp,
        next_cursor_fingerprint=next_fp,
    )
    assert delivery == delivery_again
    assert delivery not in batch_fp


@pytest.mark.unit
def test_run_id_derived_from_operation_identity() -> None:
    first = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
    )
    second = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-1",
    )
    other = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-2",
    )
    assert first == second
    assert first != other
