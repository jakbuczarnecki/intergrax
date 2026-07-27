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
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncCheckpoint
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    InMemoryCheckpointRepository,
    InMemoryLeaseRepository,
    InMemoryRemoteItemStateRepository,
    RecordingBindingService,
    RecordingFacade,
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
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=resolved_binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=lease,
        checkpoint_repository=checkpoint,
        item_state_repository=state,
        sink=sink,
        lease_ttl_seconds=30,
    )
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_corrupt_state_maps_to_non_retryable() -> None:
    state = InMemoryRemoteItemStateRepository(
        apply_error=KnowledgeSyncCorruptState("bad state")
    )
    coordinator, *_ = _build(state=state)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.sync_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False


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
    )
    await coordinator.reconcile_once(binding_id="binding-1")
    assert checkpoint.commit_calls[0]["expected_previous"] == previous
