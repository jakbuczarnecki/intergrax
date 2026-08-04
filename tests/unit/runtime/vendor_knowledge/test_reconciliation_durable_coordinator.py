# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable reconciliation coordinator, failure-window and fingerprint tests."""

from __future__ import annotations

from dataclasses import dataclass

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
    KnowledgeReconciliationRun,
    KnowledgeReconciliationRunCollecting,
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalizing_checkpoint_read_backend_failure_is_retryable() -> None:
    from datetime import datetime, timezone

    from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor
    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeReconciliationRunFinalizing,
        KnowledgeSyncCheckpoint,
        knowledge_sync_checkpoint_fingerprint_sha256,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )

    operation_id = "op-finalizing-read-fail"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    intended = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="final"),
    )
    finalizing = KnowledgeReconciliationRunFinalizing(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id=run_id,
        record_version=2,
        created_at=now,
        updated_at=now,
        applied_page_count=1,
        last_applied_delivery_id="a" * 64,
        last_applied_parent_delivery_id=None,
        intended_final_completed_checkpoint=intended,
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            intended
        ),
        expected_previous_completed_checkpoint=None,
        final_delivery_id="a" * 64,
        prepared_batch_payload_fingerprint="b" * 64,
    )
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.get_error = RuntimeError("backend down")
    coordinator, _, _, _, runs, _, _ = _durable_coordinator(checkpoint=checkpoint)
    runs.runs[("tenant-1", "binding-1")] = finalizing
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert runs.runs[("tenant-1", "binding-1")].phase is (
        KnowledgeReconciliationRunPhase.FINALIZING
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalizing_checkpoint_read_corruption_enters_recovery() -> None:
    from datetime import datetime, timezone

    from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor
    from intergrax.runtime.vendor_knowledge.sync_contracts import (
        KnowledgeSyncCorruptState,
    )
    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeReconciliationRunFinalizing,
        KnowledgeSyncCheckpoint,
        knowledge_sync_checkpoint_fingerprint_sha256,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )

    operation_id = "op-finalizing-read-corrupt"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    intended = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="final"),
    )
    finalizing = KnowledgeReconciliationRunFinalizing(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id=run_id,
        record_version=2,
        created_at=now,
        updated_at=now,
        applied_page_count=1,
        last_applied_delivery_id="a" * 64,
        last_applied_parent_delivery_id=None,
        intended_final_completed_checkpoint=intended,
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            intended
        ),
        expected_previous_completed_checkpoint=None,
        final_delivery_id="a" * 64,
        prepared_batch_payload_fingerprint="b" * 64,
    )
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.get_error = KnowledgeSyncCorruptState("corrupt checkpoint")
    coordinator, _, _, _, runs, _, _ = _durable_coordinator(checkpoint=checkpoint)
    runs.runs[("tenant-1", "binding-1")] = finalizing
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalizing_checkpoint_commit_backend_failure_is_retryable() -> None:
    from datetime import datetime, timezone

    from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor
    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeReconciliationRunFinalizing,
        KnowledgeSyncCheckpoint,
        knowledge_sync_checkpoint_fingerprint_sha256,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )

    operation_id = "op-finalizing-commit-fail"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    intended = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="final"),
    )
    finalizing = KnowledgeReconciliationRunFinalizing(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id=run_id,
        record_version=2,
        created_at=now,
        updated_at=now,
        applied_page_count=1,
        last_applied_delivery_id="a" * 64,
        last_applied_parent_delivery_id=None,
        intended_final_completed_checkpoint=intended,
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            intended
        ),
        expected_previous_completed_checkpoint=None,
        final_delivery_id="a" * 64,
        prepared_batch_payload_fingerprint="b" * 64,
    )
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.commit_error = RuntimeError("backend down")
    coordinator, _, _, _, runs, _, _ = _durable_coordinator(checkpoint=checkpoint)
    runs.runs[("tenant-1", "binding-1")] = finalizing
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert runs.runs[("tenant-1", "binding-1")].phase is (
        KnowledgeReconciliationRunPhase.FINALIZING
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_finalizing_checkpoint_commit_corruption_enters_recovery() -> None:
    from datetime import datetime, timezone

    from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor
    from intergrax.runtime.vendor_knowledge.sync_contracts import (
        KnowledgeSyncCorruptState,
    )
    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeReconciliationRunFinalizing,
        KnowledgeSyncCheckpoint,
        knowledge_sync_checkpoint_fingerprint_sha256,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )

    operation_id = "op-finalizing-commit-corrupt"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    intended = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="final"),
    )
    finalizing = KnowledgeReconciliationRunFinalizing(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id=run_id,
        record_version=2,
        created_at=now,
        updated_at=now,
        applied_page_count=1,
        last_applied_delivery_id="a" * 64,
        last_applied_parent_delivery_id=None,
        intended_final_completed_checkpoint=intended,
        intended_final_checkpoint_fingerprint=knowledge_sync_checkpoint_fingerprint_sha256(
            intended
        ),
        expected_previous_completed_checkpoint=None,
        final_delivery_id="a" * 64,
        prepared_batch_payload_fingerprint="b" * 64,
    )
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.commit_error = KnowledgeSyncCorruptState("corrupt checkpoint commit")
    coordinator, _, _, _, runs, _, _ = _durable_coordinator(checkpoint=checkpoint)
    runs.runs[("tenant-1", "binding-1")] = finalizing
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


@dataclass
class _FaultyRunRepository(InMemoryReconciliationRunRepository):
    get_error: Exception | None = None
    create_error: Exception | None = None
    cas_replace_error: Exception | None = None
    cas_recovery_error: Exception | None = None
    cas_supersede_error: Exception | None = None

    def get(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeReconciliationRun | None:
        if self.get_error is not None:
            raise self.get_error
        return super().get(tenant_id=tenant_id, binding_id=binding_id)

    def create_initial_run(self, run: KnowledgeReconciliationRun) -> None:
        if self.create_error is not None:
            raise self.create_error
        super().create_initial_run(run)

    def cas_replace(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
    ) -> None:
        if self.cas_replace_error is not None:
            raise self.cas_replace_error
        super().cas_replace(expected=expected, replacement=replacement)

    def cas_recovery(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRun,
    ) -> None:
        if self.cas_recovery_error is not None:
            raise self.cas_recovery_error
        super().cas_recovery(expected=expected, replacement=replacement)

    def cas_supersede_terminal(
        self,
        *,
        expected: KnowledgeReconciliationRun,
        replacement: KnowledgeReconciliationRunCollecting,
    ) -> None:
        if self.cas_supersede_error is not None:
            raise self.cas_supersede_error
        super().cas_supersede_terminal(expected=expected, replacement=replacement)


@dataclass
class _FaultyCandidateInventory(InMemoryCandidateInventoryRepository):
    list_error: Exception | None = None

    def list_active_remote_ids(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        binding_configuration_version: int,
        limit: int,
    ) -> tuple[str, ...]:
        if self.list_error is not None:
            raise self.list_error
        return super().list_active_remote_ids(
            tenant_id=tenant_id,
            binding_id=binding_id,
            binding_configuration_version=binding_configuration_version,
            limit=limit,
        )


def _coordinator_with_run_repo(
    runs: _FaultyRunRepository,
) -> tuple[VendorKnowledgeSyncCoordinator, RecordingFacade]:
    binding = make_binding()
    facade = RecordingFacade()
    state = InMemoryRemoteItemStateRepository()
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=InMemoryLeaseRepository(),
        checkpoint_repository=InMemoryCheckpointRepository(),
        item_state_repository=state,
        sink=IdempotentRecordingSink(),
        lease_ttl_seconds=30,
        reconciliation_run_repository=runs,
        candidate_inventory_repository=InMemoryCandidateInventoryRepository(
            state_repository=state,
        ),
        sink_receipt_inspector=RecordingSinkReceiptInspector(),
    )
    return coordinator, facade


def _assert_dependency_unavailable_no_provider_read(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
    facade: RecordingFacade,
) -> None:
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert facade.read_calls == []
    assert "backend down" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_repository_get_backend_failure_is_retryable() -> None:
    runs = _FaultyRunRepository(get_error=RuntimeError("backend down"))
    coordinator, facade = _coordinator_with_run_repo(runs)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-run-get-fail",
        )
    _assert_dependency_unavailable_no_provider_read(exc_info, facade)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_candidate_inventory_backend_failure_is_retryable() -> None:
    state = InMemoryRemoteItemStateRepository()
    inventory = _FaultyCandidateInventory(
        state_repository=state,
        list_error=RuntimeError("backend down"),
    )
    binding = make_binding()
    facade = RecordingFacade()
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=InMemoryLeaseRepository(),
        checkpoint_repository=InMemoryCheckpointRepository(),
        item_state_repository=state,
        sink=IdempotentRecordingSink(),
        lease_ttl_seconds=30,
        reconciliation_run_repository=InMemoryReconciliationRunRepository(),
        candidate_inventory_repository=inventory,
        sink_receipt_inspector=RecordingSinkReceiptInspector(),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-inventory-fail",
        )
    _assert_dependency_unavailable_no_provider_read(exc_info, facade)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_initial_run_backend_failure_is_retryable() -> None:
    runs = _FaultyRunRepository(create_error=RuntimeError("backend down"))
    coordinator, facade = _coordinator_with_run_repo(runs)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-create-fail",
        )
    _assert_dependency_unavailable_no_provider_read(exc_info, facade)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cas_replace_backend_failure_is_retryable() -> None:
    from datetime import datetime, timezone

    from intergrax.runtime.vendor_knowledge.sync_models import (
        knowledge_cursor_fingerprint_sha256,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )

    operation_id = "op-cas-replace-fail"
    run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id=operation_id,
    )
    runs = _FaultyRunRepository(cas_replace_error=RuntimeError("backend down"))
    coordinator, facade = _coordinator_with_run_repo(runs)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    collecting = KnowledgeReconciliationRunCollecting(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id=run_id,
        record_version=1,
        created_at=now,
        updated_at=now,
        current_input_cursor_fingerprint=knowledge_cursor_fingerprint_sha256(None),
        remaining_candidate_remote_ids=(),
    )
    runs.runs[("tenant-1", "binding-1")] = collecting
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "backend down" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cas_recovery_backend_failure_is_retryable() -> None:
    from datetime import datetime, timezone

    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeReconciliationRecoveryCommand,
        KnowledgeReconciliationRecoveryCommandKind,
        KnowledgeReconciliationRunRecoveryRequired,
        knowledge_cursor_fingerprint_sha256,
        recovery_evidence_from_run,
    )

    runs = _FaultyRunRepository(cas_recovery_error=RuntimeError("backend down"))
    coordinator, facade = _coordinator_with_run_repo(runs)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    collecting = KnowledgeReconciliationRunCollecting(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id="run-cas-recovery",
        record_version=1,
        created_at=now,
        updated_at=now,
        current_input_cursor_fingerprint=knowledge_cursor_fingerprint_sha256(None),
        remaining_candidate_remote_ids=(),
    )
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
        updated_at=now,
        recovery_reason_code="test",
        recovery_evidence=recovery_evidence_from_run(collecting),
    )
    runs.runs[("tenant-1", "binding-1")] = recovery
    with pytest.raises(VendorKnowledgeError) as exc_info:
        coordinator.execute_reconciliation_recovery(
            KnowledgeReconciliationRecoveryCommand(
                kind=KnowledgeReconciliationRecoveryCommandKind.RESUME_EXACT,
                tenant_id="tenant-1",
                binding_id="binding-1",
                expected_run_id="run-cas-recovery",
                expected_run_record_version=2,
                expected_phase=KnowledgeReconciliationRunPhase.RECOVERY_REQUIRED,
                operator_reason_code="resume",
            )
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "backend down" not in str(exc_info.value)
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cas_supersede_terminal_backend_failure_is_retryable() -> None:
    from datetime import datetime, timezone

    from intergrax.runtime.vendor_knowledge.sync_models import (
        knowledge_cursor_fingerprint_sha256,
    )
    from intergrax.runtime.vendor_knowledge.sync_reconciliation import (
        derive_reconciliation_run_id,
    )

    operation_id = "op-supersede-fail"
    existing_run_id = derive_reconciliation_run_id(
        tenant_id="tenant-1",
        binding_id="binding-1",
        operation_id="op-existing",
    )
    runs = _FaultyRunRepository(cas_supersede_error=RuntimeError("backend down"))
    coordinator, facade = _coordinator_with_run_repo(runs)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    collecting = KnowledgeReconciliationRunCollecting(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        provider_id="example",
        source_kind="issues",
        run_id=existing_run_id,
        record_version=1,
        created_at=now,
        updated_at=now,
        current_input_cursor_fingerprint=knowledge_cursor_fingerprint_sha256(None),
        remaining_candidate_remote_ids=(),
    )
    runs.runs[("tenant-1", "binding-1")] = collecting
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id=operation_id,
        )
    _assert_dependency_unavailable_no_provider_read(exc_info, facade)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_repository_get_corruption_is_non_retryable() -> None:
    from intergrax.runtime.vendor_knowledge.sync_contracts import (
        KnowledgeSyncCorruptState,
    )

    runs = _FaultyRunRepository(get_error=KnowledgeSyncCorruptState("corrupt run row"))
    coordinator, facade = _coordinator_with_run_repo(runs)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-run-corrupt",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert facade.read_calls == []
    assert "corrupt run row" not in str(exc_info.value)


@dataclass
class _StaticCandidateInventoryRepository:
    inventory: tuple[str, ...] | list[str] | object

    def list_active_remote_ids(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        binding_configuration_version: int,
        limit: int,
    ) -> tuple[str, ...]:
        if isinstance(self.inventory, (tuple, list)):
            ordered = tuple(self.inventory)
            if len(ordered) > limit:
                return ordered[:limit]
            return ordered
        return self.inventory  # type: ignore[return-value]


@dataclass
class _TrackingRunRepository(InMemoryReconciliationRunRepository):
    create_calls: int = 0

    def create_initial_run(self, run: KnowledgeReconciliationRun) -> None:
        self.create_calls += 1
        super().create_initial_run(run)


def _coordinator_with_static_inventory(
    inventory: tuple[str, ...] | list[str] | object,
    *,
    runs: _TrackingRunRepository | None = None,
) -> tuple[
    VendorKnowledgeSyncCoordinator,
    RecordingFacade,
    InMemoryRemoteItemStateRepository,
    InMemoryCheckpointRepository,
    _TrackingRunRepository,
    IdempotentRecordingSink,
    RecordingSinkReceiptInspector,
]:
    binding = make_binding()
    facade = RecordingFacade()
    state = InMemoryRemoteItemStateRepository()
    checkpoint = InMemoryCheckpointRepository()
    runs = runs or _TrackingRunRepository()
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
        candidate_inventory_repository=_StaticCandidateInventoryRepository(
            inventory=inventory,
        ),
        sink_receipt_inspector=inspector,
    )
    return coordinator, facade, state, checkpoint, runs, sink, inspector


def _assert_candidate_rejection_no_side_effects(
    exc_info: pytest.ExceptionInfo[VendorKnowledgeError],
    *,
    expected_code: VendorKnowledgeErrorCode,
    facade: RecordingFacade,
    runs: _TrackingRunRepository,
    sink: IdempotentRecordingSink,
    state: InMemoryRemoteItemStateRepository,
    checkpoint: InMemoryCheckpointRepository,
    rejected_value: str,
) -> None:
    assert exc_info.value.code is expected_code
    assert exc_info.value.retryable is False
    assert facade.inspect_calls == []
    assert facade.read_calls == []
    assert runs.create_calls == 0
    assert runs.runs == {}
    assert sink.calls == []
    assert state.states == {}
    assert checkpoint.commit_calls == []
    if rejected_value:
        assert rejected_value not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inventory", "rejected_value"),
    [
        (("item-1", "item-1"), "item-1"),
        (("",), ""),
        (("   ",), "   "),
        ((123,), "123"),
    ],
)
async def test_malformed_candidate_inventory_structural_rejection(
    inventory: tuple[object, ...],
    rejected_value: str,
) -> None:
    coordinator, facade, state, checkpoint, runs, sink, _ = (
        _coordinator_with_static_inventory(inventory)
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-malformed-candidate",
        )
    _assert_candidate_rejection_no_side_effects(
        exc_info,
        expected_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
        facade=facade,
        runs=runs,
        sink=sink,
        state=state,
        checkpoint=checkpoint,
        rejected_value=rejected_value,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_candidate_inventory_remote_id_byte_limit_rejected() -> None:
    oversized = "x" * 3000
    coordinator, facade, state, checkpoint, runs, sink, _ = (
        _coordinator_with_static_inventory((oversized,))
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-remote-id-limit",
        )
    _assert_candidate_rejection_no_side_effects(
        exc_info,
        expected_code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
        facade=facade,
        runs=runs,
        sink=sink,
        state=state,
        checkpoint=checkpoint,
        rejected_value=oversized,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_candidate_inventory_payload_limit_rejected() -> None:
    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeReconciliationLimitPolicy,
    )

    inventory = ("x" * 40,)
    coordinator, facade, state, checkpoint, runs, sink, _ = (
        _coordinator_with_static_inventory(inventory)
    )
    assert coordinator._reconciliation_engine is not None
    coordinator._reconciliation_engine._policy = KnowledgeReconciliationLimitPolicy(
        max_reconciliation_candidate_payload_bytes=32,
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-payload-limit",
        )
    _assert_candidate_rejection_no_side_effects(
        exc_info,
        expected_code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
        facade=facade,
        runs=runs,
        sink=sink,
        state=state,
        checkpoint=checkpoint,
        rejected_value=inventory[0],
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_candidate_inventory_count_limit_rejected() -> None:
    from intergrax.runtime.vendor_knowledge.sync_models import (
        KnowledgeReconciliationLimitPolicy,
    )

    inventory = ("item-a", "item-b")
    coordinator, facade, state, checkpoint, runs, sink, _ = (
        _coordinator_with_static_inventory(inventory)
    )
    assert coordinator._reconciliation_engine is not None
    coordinator._reconciliation_engine._policy = KnowledgeReconciliationLimitPolicy(
        max_reconciliation_candidate_count=1,
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-count-limit",
        )
    _assert_candidate_rejection_no_side_effects(
        exc_info,
        expected_code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
        facade=facade,
        runs=runs,
        sink=sink,
        state=state,
        checkpoint=checkpoint,
        rejected_value="item-a",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_candidate_inventory_corrupt_state_is_non_retryable() -> None:
    from intergrax.runtime.vendor_knowledge.sync_contracts import (
        KnowledgeSyncCorruptState,
    )

    inventory = _FaultyCandidateInventory(
        state_repository=InMemoryRemoteItemStateRepository(),
        list_error=KnowledgeSyncCorruptState("corrupt inventory row"),
    )
    binding = make_binding()
    facade = RecordingFacade()
    state = InMemoryRemoteItemStateRepository()
    runs = _TrackingRunRepository()
    sink = IdempotentRecordingSink()
    checkpoint = InMemoryCheckpointRepository()
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
        candidate_inventory_repository=inventory,
        sink_receipt_inspector=RecordingSinkReceiptInspector(),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-inventory-corrupt",
        )
    _assert_candidate_rejection_no_side_effects(
        exc_info,
        expected_code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
        facade=facade,
        runs=runs,
        sink=sink,
        state=state,
        checkpoint=checkpoint,
        rejected_value="corrupt inventory row",
    )
