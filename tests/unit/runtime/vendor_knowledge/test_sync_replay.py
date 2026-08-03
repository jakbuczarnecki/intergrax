# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for at-least-once replay and durable ordering semantics."""

from __future__ import annotations

import pytest

from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import KnowledgeCursor
from intergrax.runtime.vendor_knowledge.sync_contracts import KnowledgeSyncCorruptState
from intergrax.runtime.vendor_knowledge.sync_coordinator import (
    VendorKnowledgeSyncCoordinator,
)
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncCheckpoint
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


def _build(
    *,
    binding=None,
    facade=None,
    lease=None,
    checkpoint=None,
    state=None,
    sink=None,
    durable: bool = False,
) -> tuple[
    VendorKnowledgeSyncCoordinator,
    RecordingFacade,
    InMemoryCheckpointRepository,
    InMemoryRemoteItemStateRepository,
    IdempotentRecordingSink,
]:
    resolved_binding = binding or make_binding()
    facade = facade or RecordingFacade(
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="cp1"))
    )
    lease = lease or InMemoryLeaseRepository()
    checkpoint = checkpoint or InMemoryCheckpointRepository()
    state = state or InMemoryRemoteItemStateRepository()
    sink = sink or IdempotentRecordingSink()
    coordinator_kwargs: dict[str, object] = {
        "tenant_id": "tenant-1",
        "owner_id": "owner-1",
        "binding_service": RecordingBindingService(binding=resolved_binding),
        "facade": facade,
        "lease_repository": lease,
        "checkpoint_repository": checkpoint,
        "item_state_repository": state,
        "sink": sink,
        "lease_ttl_seconds": 30,
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
    return coordinator, facade, checkpoint, state, sink


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delivery_id_is_deterministic() -> None:
    coordinator, _, _, _, sink = _build()
    first = await coordinator.sync_once(binding_id="binding-1")
    # Reset durable state while keeping same page inputs.
    lease = InMemoryLeaseRepository()
    checkpoint = InMemoryCheckpointRepository()
    state = InMemoryRemoteItemStateRepository()
    sink2 = IdempotentRecordingSink()
    facade = RecordingFacade(
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="cp1"))
    )
    coordinator2, *_ = _build(
        facade=facade, lease=lease, checkpoint=checkpoint, state=state, sink=sink2
    )
    second = await coordinator2.sync_once(binding_id="binding-1")
    assert first.delivery_id == second.delivery_id
    assert sink.calls[0].delivery_id == first.delivery_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cursor_change_changes_delivery_id() -> None:
    facade = RecordingFacade(
        pages_by_cursor={
            None: make_page(proposed_checkpoint=KnowledgeCursor(value="cp1")),
            "cp1": make_page(
                changes=(make_change(remote_id="item-2"),),
                proposed_checkpoint=KnowledgeCursor(value="cp2"),
            ),
        }
    )
    coordinator, _, checkpoint, _, _ = _build(facade=facade)
    first = await coordinator.sync_once(binding_id="binding-1")
    second = await coordinator.sync_once(binding_id="binding-1")
    assert first.delivery_id != second.delivery_id
    assert checkpoint.checkpoints[("tenant-1", "binding-1")].cursor.value == "cp2"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_binding_configuration_change_changes_delivery_id() -> None:
    page = make_page(proposed_checkpoint=KnowledgeCursor(value="cp1"))
    coordinator_v1, _, _, _, _ = _build(
        binding=make_binding(configuration_version=1),
        facade=RecordingFacade(default_page=page),
    )
    result_v1 = await coordinator_v1.sync_once(binding_id="binding-1")
    coordinator_v2, _, _, _, _ = _build(
        binding=make_binding(configuration_version=2),
        facade=RecordingFacade(default_page=page),
    )
    result_v2 = await coordinator_v2.sync_once(binding_id="binding-1")
    assert result_v1.delivery_id != result_v2.delivery_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_page_change_changes_delivery_id() -> None:
    coordinator_a, _, _, _, _ = _build(
        facade=RecordingFacade(
            default_page=make_page(
                changes=(make_change(remote_id="item-a"),),
                proposed_checkpoint=KnowledgeCursor(value="cp1"),
            )
        )
    )
    coordinator_b, _, _, _, _ = _build(
        facade=RecordingFacade(
            default_page=make_page(
                changes=(make_change(remote_id="item-b"),),
                proposed_checkpoint=KnowledgeCursor(value="cp1"),
            )
        )
    )
    a = await coordinator_a.sync_once(binding_id="binding-1")
    b = await coordinator_b.sync_once(binding_id="binding-1")
    assert a.delivery_id != b.delivery_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sink_failure_does_not_write_state_or_checkpoint() -> None:
    sink = IdempotentRecordingSink(fail_times=1)
    checkpoint = InMemoryCheckpointRepository()
    state = InMemoryRemoteItemStateRepository()
    coordinator, *_ = _build(sink=sink, checkpoint=checkpoint, state=state)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert state.apply_calls == []
    assert checkpoint.commit_calls == []
    assert checkpoint.checkpoints == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_state_failure_does_not_write_checkpoint() -> None:
    state = InMemoryRemoteItemStateRepository(apply_error=RuntimeError("state boom"))
    checkpoint = InMemoryCheckpointRepository()
    sink = IdempotentRecordingSink()
    coordinator, *_ = _build(sink=sink, checkpoint=checkpoint, state=state)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert len(sink.calls) == 1
    assert checkpoint.commit_calls == []
    assert checkpoint.checkpoints == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_checkpoint_failure_is_retryable() -> None:
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.fail_commit_times = 1
    coordinator, *_ = _build(checkpoint=checkpoint)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_after_checkpoint_failure_retry_same_delivery_id() -> None:
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.fail_commit_times = 1
    state = InMemoryRemoteItemStateRepository()
    sink = IdempotentRecordingSink()
    facade = RecordingFacade(
        default_page=make_page(proposed_checkpoint=KnowledgeCursor(value="cp1"))
    )
    coordinator, *_ = _build(
        facade=facade, checkpoint=checkpoint, state=state, sink=sink
    )
    with pytest.raises(VendorKnowledgeError):
        await coordinator.sync_once(binding_id="binding-1")
    first_delivery = sink.calls[0].delivery_id
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.delivery_id == first_delivery
    assert sink.calls[1].delivery_id == first_delivery


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idempotent_sink_records_one_durable_delivery() -> None:
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.fail_commit_times = 1
    sink = IdempotentRecordingSink()
    coordinator, *_ = _build(checkpoint=checkpoint, sink=sink)
    with pytest.raises(VendorKnowledgeError):
        await coordinator.sync_once(binding_id="binding-1")
    await coordinator.sync_once(binding_id="binding-1")
    assert len(sink.calls) == 2
    assert sink.durable_delivery_ids == [sink.calls[0].delivery_id]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_remote_state_may_be_reapplied() -> None:
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.fail_commit_times = 1
    state = InMemoryRemoteItemStateRepository()
    coordinator, *_ = _build(checkpoint=checkpoint, state=state)
    with pytest.raises(VendorKnowledgeError):
        await coordinator.sync_once(binding_id="binding-1")
    await coordinator.sync_once(binding_id="binding-1")
    assert len(state.apply_calls) == 2
    assert state.apply_calls[0]["delivery_id"] == state.apply_calls[1]["delivery_id"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_second_attempt_can_commit_checkpoint() -> None:
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.fail_commit_times = 1
    coordinator, *_ = _build(checkpoint=checkpoint)
    with pytest.raises(VendorKnowledgeError):
        await coordinator.sync_once(binding_id="binding-1")
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.checkpoint_advanced is True
    assert ("tenant-1", "binding-1") in checkpoint.checkpoints


@pytest.mark.unit
@pytest.mark.asyncio
async def test_facade_vendor_knowledge_error_preserves_retryable() -> None:
    facade = RecordingFacade(
        read_error=VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
            safe_message="denied",
            retryable=False,
        )
    )
    coordinator, *_ = _build(facade=facade)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.AUTHORIZATION_DENIED
    assert exc_info.value.retryable is False


_CORRUPT_SECRET = (
    "Authorization: Bearer secret-token "
    "https://user:password@example.test?access_token=leak"
)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "port",
    [
        "lease_acquire",
        "lease_release",
        "sink_apply",
        "remote_state_apply",
        "checkpoint_get",
        "checkpoint_commit",
    ],
)
async def test_corrupt_state_maps_to_invalid_provider_response(port: str) -> None:
    corrupt = KnowledgeSyncCorruptState(_CORRUPT_SECRET)
    lease = InMemoryLeaseRepository()
    checkpoint = InMemoryCheckpointRepository()
    state = InMemoryRemoteItemStateRepository()
    sink = IdempotentRecordingSink()
    if port == "lease_acquire":
        lease.acquire_error = corrupt
    elif port == "lease_release":
        lease.release_error = corrupt
    elif port == "sink_apply":
        sink.apply_error = corrupt
    elif port == "remote_state_apply":
        state.apply_error = corrupt
    elif port == "checkpoint_get":
        checkpoint.get_error = corrupt
    else:
        checkpoint.commit_error = corrupt

    coordinator, *_ = _build(lease=lease, checkpoint=checkpoint, state=state, sink=sink)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert error.retryable is False
    assert error.__cause__ is None
    text = str(error)
    assert "secret-token" not in text
    assert "example.test" not in text
    assert "access_token" not in text
    assert "password" not in text


@pytest.mark.unit
@pytest.mark.asyncio
async def test_operation_error_takes_precedence_over_release_corrupt() -> None:
    lease = InMemoryLeaseRepository(
        release_error=KnowledgeSyncCorruptState(_CORRUPT_SECRET)
    )
    sink = IdempotentRecordingSink(apply_error=RuntimeError("primary sink boom"))
    coordinator, *_ = _build(lease=lease, sink=sink)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert "Knowledge sync sink failed" in exc_info.value.safe_message
    assert "secret-token" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_failure_after_success_is_retryable_dependency() -> None:
    lease = InMemoryLeaseRepository(release_error=RuntimeError("release boom"))
    coordinator, *_ = _build(lease=lease)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert exc_info.value.retryable is True
    assert "release boom" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_release_corrupt_after_success_is_non_retryable() -> None:
    lease = InMemoryLeaseRepository(
        release_error=KnowledgeSyncCorruptState(_CORRUPT_SECRET)
    )
    coordinator, *_ = _build(lease=lease)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert "secret-token" not in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reconciliation_cas_uses_loaded_checkpoint_expectation() -> None:
    previous = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="old-cursor"),
    )
    checkpoint = InMemoryCheckpointRepository()
    checkpoint.checkpoints[("tenant-1", "binding-1")] = previous
    coordinator, *_ = _build(
        checkpoint=checkpoint,
        facade=RecordingFacade(
            default_page=make_page(
                proposed_checkpoint=KnowledgeCursor(value="new-cursor")
            )
        ),
        durable=True,
    )
    await coordinator.reconcile_once(binding_id="binding-1", operation_id="op-1")
    assert checkpoint.commit_calls[0]["expected_previous"] == previous


@pytest.mark.unit
@pytest.mark.asyncio
async def test_document_store_replay_after_state_crash_before_marker() -> None:
    """Sink accepted delivery; state batch interrupted; replay completes without duplicates."""
    from intergrax.integrations._shared.in_memory_document_store import (
        InMemoryDocumentStore,
    )
    from intergrax.runtime.vendor_knowledge.sync_document_store import (
        DocumentStoreKnowledgeRemoteItemStateRepository,
        DocumentStoreKnowledgeSourceLeaseRepository,
        DocumentStoreKnowledgeSyncCheckpointRepository,
    )

    store = InMemoryDocumentStore()
    lease = DocumentStoreKnowledgeSourceLeaseRepository(store)
    checkpoint = DocumentStoreKnowledgeSyncCheckpointRepository(store)
    inner_state = DocumentStoreKnowledgeRemoteItemStateRepository(store)

    class _CrashOnceState:
        def __init__(self) -> None:
            self._failed = False

        def get(self, *, tenant_id: str, binding_id: str, remote_id: str):
            return inner_state.get(
                tenant_id=tenant_id,
                binding_id=binding_id,
                remote_id=remote_id,
            )

        def apply_batch(
            self, *, tenant_id: str, binding_id: str, delivery_id: str, states
        ):
            if not self._failed:
                self._failed = True
                if states:
                    first = states[0]
                    inner_state.apply_batch(
                        tenant_id=tenant_id,
                        binding_id=binding_id,
                        delivery_id=delivery_id,
                        states=(first,),
                    )
                    # Remove marker so batch looks incomplete after crash.
                    store.delete(
                        f"vendor_knowledge.remote_item.v1:{tenant_id}:{binding_id}",
                        f"delivery:{delivery_id}",
                    )
                raise RuntimeError("crash before full state batch")
            return inner_state.apply_batch(
                tenant_id=tenant_id,
                binding_id=binding_id,
                delivery_id=delivery_id,
                states=states,
            )

    sink = IdempotentRecordingSink()
    facade = RecordingFacade(
        default_page=make_page(
            changes=(make_change(remote_id="item-1"), make_change(remote_id="item-2")),
            proposed_checkpoint=KnowledgeCursor(value="cp1"),
        )
    )
    coordinator, *_ = _build(
        facade=facade,
        lease=lease,
        checkpoint=checkpoint,
        state=_CrashOnceState(),
        sink=sink,
    )
    with pytest.raises(VendorKnowledgeError):
        await coordinator.sync_once(binding_id="binding-1")
    assert len(sink.durable_delivery_ids) == 1
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.status.value == "completed"
    assert len(sink.durable_delivery_ids) == 1
    assert sink.calls[0].delivery_id == result.delivery_id
    assert checkpoint.get(tenant_id="tenant-1", binding_id="binding-1") is not None
    assert (
        inner_state.get(
            tenant_id="tenant-1", binding_id="binding-1", remote_id="item-2"
        )
        is not None
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_document_store_replay_after_checkpoint_crash() -> None:
    """States+marker durable; checkpoint commit fails once; replay commits checkpoint."""
    from intergrax.integrations._shared.in_memory_document_store import (
        InMemoryDocumentStore,
    )
    from intergrax.runtime.vendor_knowledge.sync_document_store import (
        DocumentStoreKnowledgeRemoteItemStateRepository,
        DocumentStoreKnowledgeSourceLeaseRepository,
        DocumentStoreKnowledgeSyncCheckpointRepository,
    )

    store = InMemoryDocumentStore()
    lease = DocumentStoreKnowledgeSourceLeaseRepository(store)
    inner_checkpoint = DocumentStoreKnowledgeSyncCheckpointRepository(store)
    state = DocumentStoreKnowledgeRemoteItemStateRepository(store)

    class _FailOnceCheckpoint:
        def __init__(self) -> None:
            self._failed = False

        def get(self, *, tenant_id: str, binding_id: str):
            return inner_checkpoint.get(tenant_id=tenant_id, binding_id=binding_id)

        def commit(self, checkpoint, *, expected_previous):
            if not self._failed:
                self._failed = True
                raise RuntimeError("crash before checkpoint commit")
            return inner_checkpoint.commit(
                checkpoint,
                expected_previous=expected_previous,
            )

    sink = IdempotentRecordingSink()
    coordinator, *_ = _build(
        lease=lease,
        checkpoint=_FailOnceCheckpoint(),
        state=state,
        sink=sink,
    )
    with pytest.raises(VendorKnowledgeError):
        await coordinator.sync_once(binding_id="binding-1")
    assert len(sink.durable_delivery_ids) == 1
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.status.value == "completed"
    assert result.checkpoint_advanced is True
    assert len(sink.durable_delivery_ids) == 1
    assert (
        inner_checkpoint.get(tenant_id="tenant-1", binding_id="binding-1") is not None
    )


@pytest.mark.unit
def test_document_store_stale_checkpoint_and_lease_token() -> None:
    from intergrax.integrations._shared.in_memory_document_store import (
        InMemoryDocumentStore,
    )
    from intergrax.runtime.vendor_knowledge.sync_contracts import (
        KnowledgeSyncCheckpointConflict,
    )
    from intergrax.runtime.vendor_knowledge.sync_document_store import (
        DocumentStoreKnowledgeSourceLeaseRepository,
        DocumentStoreKnowledgeSyncCheckpointRepository,
    )

    store = InMemoryDocumentStore()
    checkpoints = DocumentStoreKnowledgeSyncCheckpointRepository(store)
    first = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="a"),
    )
    second = KnowledgeSyncCheckpoint(
        tenant_id="tenant-1",
        binding_id="binding-1",
        binding_configuration_version=1,
        cursor=KnowledgeCursor(value="b"),
    )
    checkpoints.commit(first, expected_previous=None)
    checkpoints.commit(second, expected_previous=first)
    with pytest.raises(KnowledgeSyncCheckpointConflict):
        checkpoints.commit(
            KnowledgeSyncCheckpoint(
                tenant_id="tenant-1",
                binding_id="binding-1",
                binding_configuration_version=1,
                cursor=KnowledgeCursor(value="stale"),
            ),
            expected_previous=first,
        )

    clock = {"now": 1.0}
    tokens = iter(["lease-a", "lease-probe", "lease-b", "lease-busy-probe"])
    leases = DocumentStoreKnowledgeSourceLeaseRepository(
        store,
        clock=lambda: clock["now"],
        token_factory=lambda: next(tokens),
    )
    lease_a = leases.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-a",
        ttl_seconds=5,
    )
    assert lease_a is not None
    clock["now"] = 10.0
    lease_b = leases.acquire(
        tenant_id="tenant-1",
        binding_id="binding-1",
        owner_id="owner-b",
        ttl_seconds=5,
    )
    assert lease_b is not None
    leases.release(lease=lease_a)
    assert (
        leases.acquire(
            tenant_id="tenant-1",
            binding_id="binding-1",
            owner_id="owner-c",
            ttl_seconds=5,
        )
        is None
    )
