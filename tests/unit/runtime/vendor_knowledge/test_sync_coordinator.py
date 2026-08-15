# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for VendorKnowledgeSyncCoordinator page orchestration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChangeKind,
    KnowledgeCursor,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeRemoteItemStatus,
    KnowledgeSourceLeaseToken,
    KnowledgeSyncCheckpoint,
    KnowledgeSyncMode,
    KnowledgeSyncRunStatus,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncPublicationFenceV1,
    KnowledgeSyncPublicationInProgress,
    KnowledgeSyncPublicationPermitV1,
)
from tests.unit.runtime.vendor_knowledge._fakes import (
    FakeAdapter,
    FakeIntegration,
    RecordingResolver,
)
from tests.unit.runtime.vendor_knowledge._fakes import make_page as make_facade_page
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
    make_descriptor,
    make_page,
    shared_order,
)


def _coordinator(
    *,
    binding=None,
    lease=None,
    checkpoint=None,
    state=None,
    sink=None,
    facade=None,
    binding_service=None,
    durable: bool = False,
    publication_fence_port=None,
    require_fenced_publication: bool = False,
) -> tuple[
    VendorKnowledgeSyncCoordinator,
    RecordingBindingService,
    RecordingFacade,
    InMemoryLeaseRepository,
    InMemoryCheckpointRepository,
    InMemoryRemoteItemStateRepository,
    IdempotentRecordingSink,
]:
    resolved_binding = binding or make_binding()
    binding_service = binding_service or RecordingBindingService(
        binding=resolved_binding
    )
    facade = facade or RecordingFacade()
    lease = lease or InMemoryLeaseRepository()
    checkpoint = checkpoint or InMemoryCheckpointRepository()
    state = state or InMemoryRemoteItemStateRepository()
    sink = sink or IdempotentRecordingSink()
    coordinator_kwargs: dict[str, object] = {
        "tenant_id": "tenant-1",
        "owner_id": "owner-1",
        "binding_service": binding_service,
        "facade": facade,
        "lease_repository": lease,
        "checkpoint_repository": checkpoint,
        "item_state_repository": state,
        "sink": sink,
        "lease_ttl_seconds": 30,
        "publication_fence_port": publication_fence_port,
        "require_fenced_publication": require_fenced_publication,
    }
    if durable:
        coordinator_kwargs.update(
            {
                "reconciliation_run_repository": InMemoryReconciliationRunRepository(),
                "candidate_inventory_repository": InMemoryCandidateInventoryRepository(
                    state_repository=state
                ),
                "sink_receipt_inspector": RecordingSinkReceiptInspector(),
            }
        )
    coordinator = VendorKnowledgeSyncCoordinator(**coordinator_kwargs)  # type: ignore[arg-type]
    return coordinator, binding_service, facade, lease, checkpoint, state, sink


class _MutablePublicationFencePort:
    def __init__(self, fence: KnowledgeSyncPublicationFenceV1) -> None:
        self.fence = fence
        self.permit: KnowledgeSyncPublicationPermitV1 | None = None

    def read_fence(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSyncPublicationFenceV1 | None:
        if self.fence.tenant_id != tenant_id or self.fence.binding_id != binding_id:
            return None
        return self.fence

    def acquire_publication_permit(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        expected_revision: int,
        expected_token: str,
        owner_id: str,
        ttl_seconds: int,
    ) -> KnowledgeSyncPublicationPermitV1 | None:
        if (
            self.fence.tenant_id != tenant_id
            or self.fence.binding_id != binding_id
            or self.fence.lifecycle_revision != expected_revision
            or self.fence.lifecycle_token != expected_token
            or not self.fence.enabled
            or self.fence.detached
            or self.permit is not None
        ):
            return None
        now = datetime.now(UTC)
        self.permit = KnowledgeSyncPublicationPermitV1(
            tenant_id=tenant_id,
            binding_id=binding_id,
            lifecycle_revision=expected_revision,
            lifecycle_token=expected_token,
            permit_id=f"permit-{owner_id}",
            owner_id=owner_id,
            acquired_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        return self.permit

    def release_publication_permit(
        self,
        *,
        permit: KnowledgeSyncPublicationPermitV1,
    ) -> bool:
        if self.permit is None:
            return True
        if self.permit != permit:
            return False
        self.permit = None
        return True

    def is_current_publication_permit(
        self,
        *,
        permit: KnowledgeSyncPublicationPermitV1,
    ) -> bool:
        return (
            self.permit == permit
            and self.fence.lifecycle_revision == permit.lifecycle_revision
            and self.fence.lifecycle_token == permit.lifecycle_token
            and self.fence.enabled
            and not self.fence.detached
            and permit.expires_at > datetime.now(UTC)
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lease_busy_skips_dependencies() -> None:
    lease = InMemoryLeaseRepository(force_busy=True)
    binding_service = RecordingBindingService(binding=make_binding())
    facade = RecordingFacade()
    checkpoint = InMemoryCheckpointRepository()
    state = InMemoryRemoteItemStateRepository()
    sink = IdempotentRecordingSink()
    coordinator, *_ = _coordinator(
        lease=lease,
        binding_service=binding_service,
        facade=facade,
        checkpoint=checkpoint,
        state=state,
        sink=sink,
    )
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.status is KnowledgeSyncRunStatus.LEASE_BUSY
    assert result.retryable is True
    assert binding_service.get_calls == []
    assert facade.inspect_calls == []
    assert checkpoint.get_calls == []
    assert sink.calls == []
    assert state.apply_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_disable_after_read_rejects_all_new_publication() -> None:
    initial = KnowledgeSyncPublicationFenceV1(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=1,
        lifecycle_token="token-a",
        enabled=True,
        detached=False,
    )
    disabled = initial.model_copy(
        update={
            "lifecycle_revision": 2,
            "lifecycle_token": "token-b",
            "enabled": False,
        }
    )
    fence_port = _MutablePublicationFencePort(initial)
    facade = RecordingFacade()

    def _read_page(cursor, limit):
        fence_port.fence = disabled
        return make_page(proposed_checkpoint=KnowledgeCursor(value="cp-stale"))

    facade.page_factory = _read_page
    coordinator, _, _, _, checkpoint, state, sink = _coordinator(
        facade=facade,
        publication_fence_port=fence_port,
        require_fenced_publication=True,
    )

    result = await coordinator.sync_once(binding_id="binding-1")

    assert result.status is KnowledgeSyncRunStatus.PUBLICATION_DISABLED
    assert result.delivery_id is None
    assert checkpoint.commit_calls == []
    assert state.apply_calls == []
    assert sink.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reenable_rejects_old_token_and_allows_new_coordinator() -> None:
    fence_a = KnowledgeSyncPublicationFenceV1(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=1,
        lifecycle_token="token-a",
        enabled=True,
        detached=False,
    )
    fence_b = fence_a.model_copy(
        update={"lifecycle_revision": 2, "lifecycle_token": "token-b"}
    )
    port = _MutablePublicationFencePort(fence_a)
    facade = RecordingFacade()

    def _rotate_before_publication(cursor, limit):
        port.fence = fence_b
        return make_page(proposed_checkpoint=KnowledgeCursor(value="old"))

    facade.page_factory = _rotate_before_publication
    old, binding_service, _, lease, checkpoint, state, sink = _coordinator(
        facade=facade,
        publication_fence_port=port,
        require_fenced_publication=True,
    )
    old_result = await old.sync_once(binding_id="binding-1")

    port.fence = fence_b
    new, *_ = _coordinator(
        binding_service=binding_service,
        lease=lease,
        checkpoint=checkpoint,
        state=state,
        sink=sink,
        publication_fence_port=port,
        require_fenced_publication=True,
    )
    new_result = await new.sync_once(binding_id="binding-1")

    assert old_result.status is KnowledgeSyncRunStatus.PUBLICATION_FENCE_LOST
    assert old_result.delivery_id is None
    assert new_result.status is KnowledgeSyncRunStatus.COMPLETED
    assert checkpoint.commit_calls


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lease_loss_rejects_fenced_publication() -> None:
    class _LeaseLost(InMemoryLeaseRepository):
        def is_owned(self, *, lease: KnowledgeSourceLeaseToken) -> bool:
            return False

    fence = KnowledgeSyncPublicationFenceV1(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=1,
        lifecycle_token="token-a",
        enabled=True,
        detached=False,
    )
    lease = _LeaseLost()
    coordinator, _, _, _, checkpoint, state, sink = _coordinator(
        lease=lease,
        publication_fence_port=_MutablePublicationFencePort(fence),
        require_fenced_publication=True,
    )

    result = await coordinator.sync_once(binding_id="binding-1")

    assert result.status is KnowledgeSyncRunStatus.PUBLICATION_LEASE_LOST
    assert checkpoint.commit_calls == []
    assert state.apply_calls == []
    assert sink.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_publication_permit_is_carried_through_page_and_released() -> None:
    fence = KnowledgeSyncPublicationFenceV1(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=1,
        lifecycle_token="token-a",
        enabled=True,
        detached=False,
    )
    facade = RecordingFacade(
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="cp-1"))
    )
    port = _MutablePublicationFencePort(fence)
    coordinator, _, _, _, checkpoint, state, sink = _coordinator(
        facade=facade,
        publication_fence_port=port,
        require_fenced_publication=True,
    )

    result = await coordinator.sync_once(binding_id="binding-1")

    assert result.status is KnowledgeSyncRunStatus.COMPLETED
    assert sink.calls[0].publication_permit is not None
    assert state.apply_calls[0]["publication_permit"] is not None
    assert checkpoint.commit_calls[0]["publication_permit"] is not None
    assert port.permit is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifecycle_mutation_conflicts_between_sink_and_state_stages() -> None:
    store = InMemoryDocumentStore()
    fence_repository = DocumentStoreKnowledgeSyncPublicationFenceRepository(store)
    fence = KnowledgeSyncPublicationFenceV1(
        tenant_id="tenant-1",
        binding_id="binding-1",
        lifecycle_revision=1,
        lifecycle_token="token-a",
        enabled=True,
        detached=False,
    )
    fence_repository.write_fence(fence, expected_revision=None)

    class _RacingSink:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.disable_conflict = False

        async def apply_batch(self, *, batch: object) -> None:
            self.calls.append(batch)
            try:
                fence_repository.disable(
                    tenant_id="tenant-1",
                    binding_id="binding-1",
                    lifecycle_revision=2,
                    lifecycle_token="token-b",
                    expected_revision=1,
                )
            except KnowledgeSyncPublicationInProgress:
                self.disable_conflict = True

    sink = _RacingSink()
    facade = RecordingFacade(
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="cp-1"))
    )
    coordinator, _, _, _, checkpoint, state, _ = _coordinator(
        facade=facade,
        sink=sink,  # type: ignore[arg-type]
        publication_fence_port=fence_repository,
        require_fenced_publication=True,
    )

    result = await coordinator.sync_once(binding_id="binding-1")

    assert result.status is KnowledgeSyncRunStatus.COMPLETED
    assert sink.disable_conflict is True
    assert checkpoint.commit_calls
    assert state.apply_calls
    assert fence_repository.read_fence(
        tenant_id="tenant-1", binding_id="binding-1"
    ) == fence


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lease_released_after_success() -> None:
    coordinator, _, _, lease, *_ = _coordinator()
    await coordinator.sync_once(binding_id="binding-1")
    assert len(lease.release_calls) == 1
    assert lease.held == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lease_released_after_error() -> None:
    facade = RecordingFacade(
        read_error=VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.RATE_LIMITED,
            safe_message="rate limited",
            retryable=True,
        )
    )
    coordinator, _, _, lease, *_ = _coordinator(facade=facade)
    with pytest.raises(VendorKnowledgeError):
        await coordinator.sync_once(binding_id="binding-1")
    assert len(lease.release_calls) == 1
    assert lease.held == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_sync_uses_null_cursor() -> None:
    facade = RecordingFacade()
    coordinator, *_rest = _coordinator(facade=facade)
    await coordinator.sync_once(binding_id="binding-1")
    assert facade.read_calls[0]["cursor"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_next_sync_uses_committed_checkpoint() -> None:
    cursor_a = KnowledgeCursor(value="cursor-a", version="v1")
    cursor_b = KnowledgeCursor(value="cursor-b", version="v1")
    facade = RecordingFacade(
        pages_by_cursor={
            None: make_page(
                has_more=True,
                next_cursor=cursor_a,
                proposed_checkpoint=cursor_a,
            ),
            "cursor-a": make_page(proposed_checkpoint=cursor_b),
        }
    )
    coordinator, _, _, _, checkpoint, _, sink = _coordinator(facade=facade)
    first = await coordinator.sync_once(binding_id="binding-1")
    assert first.checkpoint_advanced is True
    assert first.has_more is True
    assert len(sink.calls) == 1
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor == cursor_a
    second = await coordinator.sync_once(binding_id="binding-1")
    assert facade.read_calls[1]["cursor"] == cursor_a
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor == cursor_b
    assert second.delivery_id != first.delivery_id
    assert sink.calls[0].delivery_id != sink.calls[1].delivery_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_incremental_two_page_continuation() -> None:
    initial_cursor = KnowledgeCursor(value="sync-v1")
    cp1 = KnowledgeCursor(value="sync-v1-page-2")
    final_delta = KnowledgeCursor(value="final-delta")
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.checkpoints[("tenant-1", "binding-1")] = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=initial_cursor,
    )
    facade = RecordingFacade(
        pages_by_cursor={
            initial_cursor.value: make_page(
                changes=(make_change(remote_id="item-1"),),
                has_more=True,
                next_cursor=cp1,
                proposed_checkpoint=cp1,
            ),
            cp1.value: make_page(
                changes=(make_change(remote_id="item-2"),),
                proposed_checkpoint=final_delta,
                has_more=False,
            ),
        }
    )
    coordinator, binding_service, _, lease, _, state, sink = _coordinator(
        facade=facade,
        checkpoint=checkpoint,
    )
    first = await coordinator.sync_once(binding_id="binding-1")
    assert facade.read_calls[0]["cursor"] == initial_cursor
    assert first.has_more is True
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor == cp1
    assert len(sink.calls) == 1
    fresh_coordinator, *_ = _coordinator(
        binding_service=binding_service,
        facade=facade,
        lease=lease,
        checkpoint=checkpoint,
        state=state,
        sink=sink,
    )
    second = await fresh_coordinator.sync_once(binding_id="binding-1")
    assert facade.read_calls[1]["cursor"] == cp1
    assert second.has_more is False
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor == final_delta
    assert len(facade.read_calls) == 2
    assert len(sink.calls) == 2
    assert sink.calls[0].delivery_id != sink.calls[1].delivery_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_two_page_continuation() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    final_cp = KnowledgeCursor(value="reconcile-final")
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            reconciliation=True,
            full_inventory=False,
            incremental_changes=False,
            content_fetch=True,
            structured_content=True,
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
    coordinator, _, _, _, checkpoint, _, sink = _coordinator(
        facade=facade, durable=True
    )
    operation_id = "op-recon-flow"
    first = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    assert first.mode is KnowledgeSyncMode.RECONCILIATION
    assert first.has_more is True
    assert facade.read_calls[0]["cursor"] is None
    assert ("tenant-1", "binding-1") not in checkpoint.checkpoints
    second = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=first.delivery_id,
    )
    assert second.mode is KnowledgeSyncMode.RECONCILIATION
    assert second.has_more is False
    assert facade.read_calls[-1]["cursor"] == reconcile_cp1
    assert len(facade.read_calls) == 2
    assert len(sink.calls) == 2
    assert sink.calls[0].delivery_id != sink.calls[1].delivery_id
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor == final_cp
    assert len(checkpoint.commit_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_restart_false_without_checkpoint() -> None:
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
    coordinator, _, facade_rec, _, checkpoint, state, sink = _coordinator(
        facade=facade, durable=True
    )
    operation_id = "op-1"
    await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=False,
            operation_id=operation_id,
            trigger_delivery_id="f" * 64,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert exc_info.value.retryable is False
    assert len(facade_rec.read_calls) == 1
    assert len(sink.calls) == 1
    assert len(state.apply_calls) == 1
    assert checkpoint.commit_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_restart_false_stale_configuration() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.checkpoints[("tenant-1", "binding-1")] = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="secret-reconcile-cursor"),
    )
    binding = make_binding(configuration_version=2)
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
    coordinator, *_ = _coordinator(
        binding=binding, facade=facade, checkpoint=checkpoint, durable=True
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1", restart=False, operation_id="op-1"
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert exc_info.value.retryable is False
    assert "secret-reconcile-cursor" not in exc_info.value.safe_message
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_restart_true_with_stale_checkpoint() -> None:
    previous = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="old-config-cursor"),
    )
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.checkpoints[("tenant-1", "binding-1")] = previous
    binding = make_binding(configuration_version=2)
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            reconciliation=True, content_fetch=True
        ),
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="new-cursor")),
    )
    coordinator, *_ = _coordinator(
        binding=binding, facade=facade, checkpoint=checkpoint, durable=True
    )
    result = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id="op-1"
    )
    assert result.mode is KnowledgeSyncMode.RECONCILIATION
    assert facade.read_calls[0]["cursor"] is None
    assert len(checkpoint.commit_calls) == 1
    assert checkpoint.commit_calls[0]["expected_previous"] == previous
    assert (
        checkpoint.checkpoints[("tenant-1", "binding-1")].binding_configuration_version
        == 2
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_has_more_mismatched_next_and_proposed_checkpoint_fail_closed() -> None:
    facade = RecordingFacade(
        default_page=make_page(
            has_more=True,
            next_cursor=KnowledgeCursor(value="next-token"),
            proposed_checkpoint=KnowledgeCursor(value="other-token"),
        )
    )
    coordinator, *_, checkpoint, state, sink = _coordinator(facade=facade)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert "inconsistent" in exc_info.value.safe_message.lower()
    assert "next-token" not in exc_info.value.safe_message
    assert "other-token" not in exc_info.value.safe_message
    assert sink.calls == []
    assert state.apply_calls == []
    assert checkpoint.commit_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_has_more_with_matching_continuation_cursor_succeeds() -> None:
    continuation = KnowledgeCursor(value="shared-continuation")
    facade = RecordingFacade(
        default_page=make_page(
            has_more=True,
            next_cursor=continuation,
            proposed_checkpoint=continuation,
        )
    )
    coordinator, *_, checkpoint, _, sink = _coordinator(facade=facade)
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.has_more is True
    assert len(sink.calls) == 1
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor == continuation


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_configuration_checkpoint_requires_reconciliation() -> None:
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.checkpoints[("tenant-1", "binding-1")] = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="old"),
    )
    binding = make_binding(configuration_version=2)
    coordinator, *_ = _coordinator(binding=binding, checkpoint=checkpoint)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert exc_info.value.retryable is False
    assert "reconciliation" in exc_info.value.safe_message.lower()
    assert "old" not in exc_info.value.safe_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_uses_null_cursor() -> None:
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.checkpoints[("tenant-1", "binding-1")] = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="incremental-cursor"),
    )
    facade = RecordingFacade(
        default_page=make_page(
            proposed_checkpoint=KnowledgeCursor(value="reconciled", version="v1")
        )
    )
    coordinator, *_ = _coordinator(facade=facade, checkpoint=checkpoint, durable=True)
    await coordinator.reconcile_once(binding_id="binding-1", operation_id="op-1")
    assert facade.read_calls[0]["cursor"] is None
    assert (
        checkpoint.checkpoints[("tenant-1", "binding-1")].cursor.value == "reconciled"
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_requires_capability() -> None:
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            incremental_changes=True,
            content_fetch=True,
        )
    )
    coordinator, *_ = _coordinator(facade=facade, durable=True)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(binding_id="binding-1", operation_id="op-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert exc_info.value.retryable is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_page_size_validation() -> None:
    coordinator, *_ = _coordinator()
    with pytest.raises(ValueError, match="page_size"):
        await coordinator.sync_once(binding_id="binding-1", page_size=0)
    with pytest.raises(ValueError, match="page_size"):
        await coordinator.sync_once(binding_id="binding-1", page_size=1001)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exactly_one_page_per_call() -> None:
    continuation = KnowledgeCursor(value="cp1")
    facade = RecordingFacade(
        default_page=make_page(
            has_more=True,
            next_cursor=continuation,
            proposed_checkpoint=continuation,
        )
    )
    coordinator, *_ = _coordinator(facade=facade)
    result = await coordinator.sync_once(binding_id="binding-1")
    assert len(facade.read_calls) == 1
    assert result.has_more is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_fetches_content_and_permissions() -> None:
    facade = RecordingFacade()
    coordinator, *_ = _coordinator(facade=facade)
    await coordinator.sync_once(binding_id="binding-1")
    assert len(facade.content_calls) == 1
    assert len(facade.permissions_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_metadata_change_skips_content_and_permissions() -> None:
    facade = RecordingFacade(
        default_page=make_page(
            changes=(make_change(kind=KnowledgeChangeKind.METADATA_CHANGED),),
            proposed_checkpoint=KnowledgeCursor(value="cp1"),
        )
    )
    coordinator, *_, sink = _coordinator(facade=facade)
    await coordinator.sync_once(binding_id="binding-1")
    assert facade.content_calls == []
    assert facade.permissions_calls == []
    assert sink.calls[0].envelopes[0].content is None
    assert sink.calls[0].envelopes[0].permissions is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_permissions_change_fetches_only_permissions() -> None:
    facade = RecordingFacade(
        default_page=make_page(
            changes=(make_change(kind=KnowledgeChangeKind.PERMISSIONS_CHANGED),),
            proposed_checkpoint=KnowledgeCursor(value="cp1"),
        )
    )
    coordinator, *_ = _coordinator(facade=facade)
    await coordinator.sync_once(binding_id="binding-1")
    assert facade.content_calls == []
    assert len(facade.permissions_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deleted_and_revoked_skip_content_and_permissions() -> None:
    facade = RecordingFacade(
        default_page=make_page(
            changes=(
                make_change(kind=KnowledgeChangeKind.DELETED, remote_id="d1"),
                make_change(kind=KnowledgeChangeKind.REVOKED, remote_id="r1"),
            ),
            proposed_checkpoint=KnowledgeCursor(value="cp1"),
        )
    )
    coordinator, *_, sink = _coordinator(facade=facade)
    result = await coordinator.sync_once(binding_id="binding-1")
    assert facade.content_calls == []
    assert facade.permissions_calls == []
    assert result.tombstone_count == 2
    assert all(env.content is None for env in sink.calls[0].envelopes)
    assert all(env.permissions is None for env in sink.calls[0].envelopes)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_duplicate_remote_id_fail_closed() -> None:
    facade = RecordingFacade(
        default_page=make_page(
            changes=(
                make_change(remote_id="same"),
                make_change(remote_id="same"),
            ),
            proposed_checkpoint=KnowledgeCursor(value="cp1"),
        )
    )
    coordinator, *_, sink = _coordinator(facade=facade)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert sink.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_has_more_without_proposed_checkpoint_fail_closed() -> None:
    facade = RecordingFacade(
        default_page=make_page(
            has_more=True,
            next_cursor=KnowledgeCursor(value="next"),
            proposed_checkpoint=None,
        )
    )
    coordinator, *_, sink = _coordinator(facade=facade)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert sink.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sink_before_state_before_checkpoint() -> None:
    sink = IdempotentRecordingSink()
    state = InMemoryRemoteItemStateRepository()
    checkpoint = InMemoryCheckpointRepository()
    facade = RecordingFacade(
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="cp1"))
    )
    coordinator, *_ = _coordinator(
        facade=facade, sink=sink, state=state, checkpoint=checkpoint
    )
    await coordinator.sync_once(binding_id="binding-1")
    assert shared_order(sink, state, checkpoint) == ["sink", "state", "checkpoint"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_state_status_mapping() -> None:
    facade = RecordingFacade(
        default_page=make_page(
            changes=(
                make_change(kind=KnowledgeChangeKind.UPSERT, remote_id="a"),
                make_change(kind=KnowledgeChangeKind.DELETED, remote_id="d"),
                make_change(kind=KnowledgeChangeKind.REVOKED, remote_id="r"),
            ),
            proposed_checkpoint=KnowledgeCursor(value="cp1"),
        )
    )
    coordinator, *_, state, _sink = _coordinator(facade=facade)
    await coordinator.sync_once(binding_id="binding-1")
    assert (
        state.states[("tenant-1", "binding-1", "a")].status
        is KnowledgeRemoteItemStatus.ACTIVE
    )
    assert (
        state.states[("tenant-1", "binding-1", "d")].status
        is KnowledgeRemoteItemStatus.DELETED
    )
    assert (
        state.states[("tenant-1", "binding-1", "r")].status
        is KnowledgeRemoteItemStatus.REVOKED
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_result_counts_and_has_more() -> None:
    continuation = KnowledgeCursor(value="cp1")
    facade = RecordingFacade(
        default_page=make_page(
            changes=(
                make_change(kind=KnowledgeChangeKind.UPSERT, remote_id="a"),
                make_change(kind=KnowledgeChangeKind.DELETED, remote_id="d"),
            ),
            has_more=True,
            next_cursor=continuation,
            proposed_checkpoint=continuation,
        )
    )
    coordinator, *_ = _coordinator(facade=facade)
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.changes_count == 2
    assert result.active_count == 1
    assert result.tombstone_count == 1
    assert result.has_more is True
    assert result.mode is KnowledgeSyncMode.INCREMENTAL
    assert result.status is KnowledgeSyncRunStatus.COMPLETED


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tenant_binding_source_consistency() -> None:
    from intergrax.integrations.contracts.base import IntegrationCategory
    from intergrax.runtime.vendor_knowledge.models import (
        KnowledgeSourceRef,
        KnowledgeSourceScope,
    )

    binding = make_binding()
    bad_source = KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id="other",
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
        connection_ref="conn-1",
        scope=KnowledgeSourceScope(
            remote_scope_id="scope-1",
            remote_scope_type="project",
            safe_display_name="Example Project",
        ),
    )
    binding_service = RecordingBindingService(
        binding=binding, source_override=bad_source
    )
    coordinator, *_ = _coordinator(binding_service=binding_service)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert "conn-1" not in exc_info.value.safe_message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_raw_port_errors_do_not_leak() -> None:
    sink = IdempotentRecordingSink(
        apply_error=RuntimeError("secret path /tmp/cursor=abc")
    )
    coordinator, *_ = _coordinator(sink=sink)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert "secret path" not in str(exc_info.value)
    assert "cursor=abc" not in str(exc_info.value)
    assert exc_info.value.retryable is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_upsert_skips_content_when_unavailable() -> None:
    facade = RecordingFacade(
        default_page=make_page(
            changes=(
                make_change(
                    descriptor=make_descriptor(content_available=False),
                ),
            ),
            proposed_checkpoint=KnowledgeCursor(value="cp1"),
        )
    )
    coordinator, *_ = _coordinator(facade=facade)
    await coordinator.sync_once(binding_id="binding-1")
    assert facade.content_calls == []
    assert len(facade.permissions_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_kwargs",),
    [
        ({"tenant_id": "other-tenant"},),
        ({"binding_id": "other-binding"},),
        ({"owner_id": "other-owner"},),
    ],
)
async def test_lease_token_identity_mismatch_is_rejected(
    token_kwargs: dict[str, str],
) -> None:
    token_fields = {
        "tenant_id": "tenant-1",
        "binding_id": "binding-1",
        "owner_id": "owner-1",
        "token": "foreign-secret-token",
    }
    token_fields.update(token_kwargs)
    lease = InMemoryLeaseRepository(
        forced_token=KnowledgeSourceLeaseToken(**token_fields)
    )
    binding_service = RecordingBindingService(binding=make_binding())
    facade = RecordingFacade()
    checkpoint = InMemoryCheckpointRepository()
    state = InMemoryRemoteItemStateRepository()
    sink = IdempotentRecordingSink()
    coordinator, *_ = _coordinator(
        lease=lease,
        binding_service=binding_service,
        facade=facade,
        checkpoint=checkpoint,
        state=state,
        sink=sink,
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert "foreign-secret-token" not in str(exc_info.value)
    assert "other-tenant" not in str(exc_info.value)
    assert "other-binding" not in str(exc_info.value)
    assert "other-owner" not in str(exc_info.value)
    assert binding_service.get_calls == []
    assert facade.inspect_calls == []
    assert facade.read_calls == []
    assert checkpoint.get_calls == []
    assert sink.calls == []
    assert state.apply_calls == []
    assert lease.release_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_valid_lease_token_still_runs_full_flow() -> None:
    token = KnowledgeSourceLeaseToken(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-1",
        token="ok-token",
    )
    lease = InMemoryLeaseRepository(
        held={("tenant-1", "binding-1"): token},
        forced_token=token,
    )
    coordinator, _, facade, lease_repo, checkpoint, state, sink = _coordinator(
        lease=lease
    )
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.status is KnowledgeSyncRunStatus.COMPLETED
    assert facade.read_calls
    assert sink.calls
    assert state.apply_calls
    assert checkpoint.commit_calls
    assert len(lease_repo.release_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "checkpoint_kwargs"),
    [
        ("incremental", {"tenant_id": "other-tenant"}),
        ("incremental", {"binding_id": "other-binding"}),
        ("reconciliation", {"tenant_id": "other-tenant"}),
        ("reconciliation", {"binding_id": "other-binding"}),
    ],
)
async def test_checkpoint_identity_mismatch_rejected_before_reads(
    mode: str,
    checkpoint_kwargs: dict[str, str],
) -> None:
    fields = {
        "tenant_id": "tenant-1",
        "binding_id": "binding-1",
        "binding_configuration_version": 1,
        "cursor": KnowledgeCursor(value="secret-cursor-value"),
    }
    fields.update(checkpoint_kwargs)
    checkpoint = InMemoryCheckpointRepository(
        forced_checkpoint=KnowledgeSyncCheckpoint(**fields)
    )
    facade = RecordingFacade()
    state = InMemoryRemoteItemStateRepository()
    sink = IdempotentRecordingSink()
    coordinator, *_ = _coordinator(
        facade=facade,
        checkpoint=checkpoint,
        state=state,
        sink=sink,
        durable=(mode == "reconciliation"),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        if mode == "incremental":
            await coordinator.sync_once(binding_id="binding-1")
        else:
            await coordinator.reconcile_once(
                binding_id="binding-1", operation_id="op-1"
            )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert "secret-cursor-value" not in str(exc_info.value)
    if mode == "incremental":
        assert facade.inspect_calls == []
    assert facade.read_calls == []
    assert sink.calls == []
    assert state.apply_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_accepts_older_configuration_checkpoint_as_cas() -> None:
    previous = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="old-config-cursor"),
    )
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.checkpoints[("tenant-1", "binding-1")] = previous
    binding = make_binding(configuration_version=2)
    facade = RecordingFacade(
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="new-cursor"))
    )
    coordinator, *_ = _coordinator(
        binding=binding, facade=facade, checkpoint=checkpoint, durable=True
    )
    result = await coordinator.reconcile_once(
        binding_id="binding-1", operation_id="op-1"
    )
    assert result.mode is KnowledgeSyncMode.RECONCILIATION
    assert len(checkpoint.commit_calls) == 1
    assert checkpoint.commit_calls[0]["expected_previous"] == previous
    assert (
        checkpoint.checkpoints[("tenant-1", "binding-1")].binding_configuration_version
        == 2
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_incremental_allows_full_inventory_only() -> None:
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            full_inventory=True, content_fetch=True
        )
    )
    coordinator, *_ = _coordinator(facade=facade)
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.status is KnowledgeSyncRunStatus.COMPLETED
    assert len(facade.read_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_incremental_allows_incremental_changes_only() -> None:
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            incremental_changes=True, content_fetch=True
        )
    )
    coordinator, *_ = _coordinator(facade=facade)
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.status is KnowledgeSyncRunStatus.COMPLETED
    assert len(facade.read_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_first_incremental_rejects_reconciliation_only_capability() -> None:
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            reconciliation=True, content_fetch=True
        )
    )
    coordinator, *_ = _coordinator(facade=facade)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert exc_info.value.retryable is False
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_subsequent_incremental_requires_incremental_changes() -> None:
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.checkpoints[("tenant-1", "binding-1")] = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="cursor-a"),
    )
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            full_inventory=True, content_fetch=True
        ),
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="cursor-b")),
    )
    coordinator, *_ = _coordinator(facade=facade, checkpoint=checkpoint)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_allows_full_inventory_without_reconciliation_flag() -> (
    None
):
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            full_inventory=True, content_fetch=True
        )
    )
    coordinator, *_ = _coordinator(facade=facade, durable=True)
    result = await coordinator.reconcile_once(
        binding_id="binding-1", operation_id="op-1"
    )
    assert result.mode is KnowledgeSyncMode.RECONCILIATION
    assert len(facade.read_calls) >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_allows_reconciliation_without_full_inventory() -> None:
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(
            reconciliation=True, content_fetch=True
        )
    )
    coordinator, *_ = _coordinator(facade=facade, durable=True)
    result = await coordinator.reconcile_once(
        binding_id="binding-1", operation_id="op-1"
    )
    assert result.mode is KnowledgeSyncMode.RECONCILIATION
    assert len(facade.read_calls) >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_capability_stops_before_read_page() -> None:
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(incremental_changes=True)
    )
    coordinator, *_ = _coordinator(facade=facade, durable=True)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(binding_id="binding-1", operation_id="op-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_coordinator_with_production_facade_reconciliation_two_pages() -> None:
    from intergrax.runtime.vendor_knowledge.models import KnowledgePage

    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    final_cp = KnowledgeCursor(value="reconcile-final")
    first_base = make_facade_page(remote_id="item-1")
    second_base = make_facade_page(remote_id="item-2")
    pages_by_cursor = {
        None: KnowledgePage(
            changes=first_base.changes,
            next_cursor=reconcile_cp1,
            proposed_checkpoint=reconcile_cp1,
            has_more=True,
        ),
        "reconcile-cp-1": KnowledgePage(
            changes=second_base.changes,
            next_cursor=None,
            proposed_checkpoint=final_cp,
            has_more=False,
        ),
    }
    adapter = FakeAdapter(
        capabilities=KnowledgeAdapterCapabilities(
            reconciliation=True,
            full_inventory=False,
            incremental_changes=False,
            content_fetch=True,
            structured_content=True,
        ),
        pages_by_cursor=pages_by_cursor,
    )
    registry = KnowledgeAdapterRegistry()
    registry.register(adapter)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=RecordingResolver(integration=FakeIntegration()),
        adapter_registry=registry,
    )
    sink = IdempotentRecordingSink()
    state = InMemoryRemoteItemStateRepository()
    checkpoint = InMemoryCheckpointRepository()
    from tests.unit.runtime.vendor_knowledge._sync_fakes import (
        InMemoryCandidateInventoryRepository,
        InMemoryReconciliationRunRepository,
        RecordingSinkReceiptInspector,
    )

    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=make_binding()),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=InMemoryLeaseRepository(),
        checkpoint_repository=checkpoint,
        item_state_repository=state,
        sink=sink,
        lease_ttl_seconds=30,
        reconciliation_run_repository=InMemoryReconciliationRunRepository(),
        candidate_inventory_repository=InMemoryCandidateInventoryRepository(
            state_repository=state
        ),
        sink_receipt_inspector=RecordingSinkReceiptInspector(),
    )
    operation_id = "op-prod"
    first = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    second = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=first.delivery_id,
    )
    assert first.mode is KnowledgeSyncMode.RECONCILIATION
    assert second.mode is KnowledgeSyncMode.RECONCILIATION
    assert second.has_more is False
    assert adapter.read_calls[-1]["cursor"] == reconcile_cp1
    assert len(sink.calls) == 2
    assert len(state.apply_calls) == 2
    assert len(checkpoint.commit_calls) == 1
    assert sink.order == ["sink", "sink"]
    assert state.order == ["state", "state"]
    assert checkpoint.order == ["checkpoint"]
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor == final_cp
    assert not isinstance(facade, RecordingFacade)
