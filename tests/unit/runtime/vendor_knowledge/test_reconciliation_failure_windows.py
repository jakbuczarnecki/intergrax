# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic failure-window tests for durable reconciliation."""

from __future__ import annotations

import pytest

from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeCursor,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeReconciliationRunPhase,
)
from tests.unit.runtime.vendor_knowledge.test_reconciliation_durable_coordinator import (
    _durable_coordinator,
)
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    InMemoryLeaseRepository,
    make_change,
    make_page,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_run_repository_fails_before_provider_read() -> None:
    coordinator, facade, *_ = _durable_coordinator()
    coordinator._reconciliation_run_repository = None  # type: ignore[attr-defined]
    coordinator._reconciliation_engine = None
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(binding_id="binding-1", operation_id="op-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_operation_id_rejected() -> None:
    coordinator, facade, *_ = _durable_coordinator()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(binding_id="binding-1")
    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_incremental_preserved_when_reconciliation_unconfigured() -> None:
    coordinator, facade, *_ = _durable_coordinator()
    coordinator._reconciliation_run_repository = None  # type: ignore[attr-defined]
    coordinator._candidate_inventory_repository = None  # type: ignore[attr-defined]
    coordinator._reconciliation_engine = None
    result = await coordinator.sync_once(binding_id="binding-1")
    assert result.retryable is False
    assert len(facade.read_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_continuation_lineage_retry_after_page_applied() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    from tests.unit.runtime.vendor_knowledge._sync_fakes import RecordingFacade

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
            "reconcile-cp-1": make_page(
                changes=(make_change(remote_id="item-2"),),
                proposed_checkpoint=KnowledgeCursor(value="final"),
                has_more=False,
            ),
        },
    )
    coordinator, _, _, _, runs, _, _ = _durable_coordinator(facade=facade)
    operation_id = "op-lineage"
    first = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    run_after_first = runs.runs[("tenant-1", "binding-1")]
    assert run_after_first.phase is KnowledgeReconciliationRunPhase.COLLECTING
    assert run_after_first.applied_page_count == 1
    retry = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=first.delivery_id,
    )
    assert retry.has_more is False
    assert runs.runs[("tenant-1", "binding-1")].applied_page_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_older_trigger_rejected_without_provider_read() -> None:
    coordinator, facade, _, _, runs, _, _ = _durable_coordinator()
    operation_id = "op-stale-lineage"
    await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    run = runs.runs[("tenant-1", "binding-1")]
    stale = run.last_applied_parent_delivery_id or "0" * 64
    reads_before = len(facade.read_calls)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=False,
            operation_id=operation_id,
            trigger_delivery_id=stale,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert len(facade.read_calls) == reads_before


@pytest.mark.unit
@pytest.mark.asyncio
async def test_operation_failure_not_masked_by_lease_release_failure() -> None:
    from tests.unit.runtime.vendor_knowledge._sync_fakes import RecordingFacade

    lease = InMemoryLeaseRepository(release_error=RuntimeError("release failed"))
    facade = RecordingFacade(
        capabilities=KnowledgeAdapterCapabilities(incremental_changes=True)
    )
    coordinator, _, _, _, _, _, _ = _durable_coordinator(facade=facade)
    coordinator._lease_repository = lease  # type: ignore[attr-defined]
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-lease",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY
    assert len(lease.release_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_job_retry_after_page_applied_is_idempotent() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    reconcile_cp2 = KnowledgeCursor(value="reconcile-cp-2")
    from tests.unit.runtime.vendor_knowledge._sync_fakes import RecordingFacade

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
            "reconcile-cp-1": make_page(
                changes=(make_change(remote_id="item-2"),),
                has_more=True,
                next_cursor=reconcile_cp2,
                proposed_checkpoint=reconcile_cp2,
            ),
            "reconcile-cp-2": make_page(
                changes=(make_change(remote_id="item-3"),),
                proposed_checkpoint=KnowledgeCursor(value="final"),
                has_more=False,
            ),
        },
    )
    coordinator, _, state, _, runs, sink, _ = _durable_coordinator(facade=facade)
    operation_id = "op-same-job"
    first = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    second = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=first.delivery_id,
    )
    run_after_second = runs.runs[("tenant-1", "binding-1")]
    assert run_after_second.phase is KnowledgeReconciliationRunPhase.COLLECTING
    assert run_after_second.applied_page_count == 2
    parent_trigger = first.delivery_id
    reads_before = len(facade.read_calls)
    sink_before = len(sink.calls)
    state_before = len(getattr(state, "states", {}))
    version_before = run_after_second.record_version
    retry = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=parent_trigger,
    )
    run_after_retry = runs.runs[("tenant-1", "binding-1")]
    assert retry.delivery_id == run_after_second.last_applied_delivery_id
    assert retry.delivery_id == second.delivery_id
    assert retry.has_more is True
    assert retry.changes_count == 0
    assert len(facade.read_calls) == reads_before
    assert len(sink.calls) == sink_before
    if hasattr(state, "states"):
        assert len(state.states) == state_before
    assert run_after_retry.record_version == version_before
    assert run_after_retry.applied_page_count == run_after_second.applied_page_count


@pytest.mark.unit
@pytest.mark.asyncio
async def test_completed_run_retry_returns_final_result_without_mutation() -> None:
    coordinator, facade, _, checkpoint, runs, sink, state = _durable_coordinator()
    operation_id = "op-completed-retry"
    completed = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    run = runs.runs[("tenant-1", "binding-1")]
    assert run.phase is KnowledgeReconciliationRunPhase.COMPLETED
    assert completed.checkpoint_advanced is True
    assert completed.has_more is False
    assert completed.delivery_id == run.final_delivery_id
    reads_before = len(facade.read_calls)
    sink_before = len(sink.calls)
    state_before = len(getattr(state, "states", {}))
    checkpoint_before = len(checkpoint.commit_calls)
    version_before = run.record_version
    retry = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=True,
        operation_id=operation_id,
    )
    assert retry.delivery_id == run.final_delivery_id
    assert retry.checkpoint_advanced is True
    assert retry.has_more is False
    assert retry.changes_count == 0
    assert len(facade.read_calls) == reads_before
    assert len(sink.calls) == sink_before
    if hasattr(state, "states"):
        assert len(state.states) == state_before
    assert len(checkpoint.commit_calls) == checkpoint_before
    assert runs.runs[("tenant-1", "binding-1")].record_version == version_before


@pytest.mark.unit
@pytest.mark.asyncio
async def test_initial_job_replay_after_first_page_is_idempotent() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    from tests.unit.runtime.vendor_knowledge._sync_fakes import RecordingFacade

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
            "reconcile-cp-1": make_page(
                changes=(make_change(remote_id="item-2"),),
                proposed_checkpoint=KnowledgeCursor(value="final"),
                has_more=False,
            ),
        },
    )
    coordinator, _, state, checkpoint, runs, sink, _ = _durable_coordinator(
        facade=facade
    )
    operation_id = "op-initial-replay"
    first = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    run_after_first = runs.runs[("tenant-1", "binding-1")]
    assert run_after_first.phase is KnowledgeReconciliationRunPhase.COLLECTING
    assert run_after_first.applied_page_count == 1
    reads_before = len(facade.read_calls)
    sink_before = len(sink.calls)
    state_before = len(getattr(state, "states", {}))
    checkpoint_before = len(checkpoint.commit_calls)
    version_before = run_after_first.record_version
    retry = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=True,
        operation_id=operation_id,
    )
    assert retry.delivery_id == first.delivery_id
    assert retry.has_more is True
    assert retry.changes_count == 0
    assert len(facade.read_calls) == reads_before
    assert len(sink.calls) == sink_before
    if hasattr(state, "states"):
        assert len(state.states) == state_before
    assert len(checkpoint.commit_calls) == checkpoint_before
    assert runs.runs[("tenant-1", "binding-1")].record_version == version_before
    assert runs.runs[("tenant-1", "binding-1")].applied_page_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_initial_job_rejected_after_page_two_lineage() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    reconcile_cp2 = KnowledgeCursor(value="reconcile-cp-2")
    from tests.unit.runtime.vendor_knowledge._sync_fakes import RecordingFacade

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
            "reconcile-cp-1": make_page(
                changes=(make_change(remote_id="item-2"),),
                has_more=True,
                next_cursor=reconcile_cp2,
                proposed_checkpoint=reconcile_cp2,
            ),
        },
    )
    coordinator, facade_ref, _, _, runs, _, _ = _durable_coordinator(facade=facade)
    operation_id = "op-lineage-reject"
    first = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=first.delivery_id,
    )
    assert runs.runs[("tenant-1", "binding-1")].applied_page_count == 2
    reads_before = len(facade_ref.read_calls)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id=operation_id,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert len(facade_ref.read_calls) == reads_before


@pytest.mark.unit
@pytest.mark.asyncio
async def test_restart_false_without_trigger_rejected() -> None:
    coordinator, facade, _, _, _, _, _ = _durable_coordinator()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=False,
            operation_id="op-no-trigger",
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_restart_true_with_trigger_rejected() -> None:
    coordinator, facade, _, _, _, _, _ = _durable_coordinator()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(
            binding_id="binding-1",
            restart=True,
            operation_id="op-bad-trigger",
            trigger_delivery_id="a" * 64,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert facade.read_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multi_page_completed_replay_reports_committed_checkpoint() -> None:
    reconcile_cp1 = KnowledgeCursor(value="reconcile-cp-1")
    from tests.unit.runtime.vendor_knowledge._sync_fakes import RecordingFacade

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
            "reconcile-cp-1": make_page(
                changes=(make_change(remote_id="item-2"),),
                proposed_checkpoint=KnowledgeCursor(value="final"),
                has_more=False,
            ),
        },
    )
    coordinator, facade_ref, _, checkpoint, runs, sink, state = _durable_coordinator(
        facade=facade
    )
    operation_id = "op-multi-complete"
    first = await coordinator.reconcile_once(
        binding_id="binding-1", restart=True, operation_id=operation_id
    )
    completed = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=first.delivery_id,
    )
    run = runs.runs[("tenant-1", "binding-1")]
    assert run.phase is KnowledgeReconciliationRunPhase.COMPLETED
    assert completed.checkpoint_advanced is True
    reads_before = len(facade_ref.read_calls)
    sink_before = len(sink.calls)
    state_before = len(getattr(state, "states", {}))
    checkpoint_before = len(checkpoint.commit_calls)
    version_before = run.record_version
    parent = run.last_applied_parent_delivery_id
    assert parent is not None
    retry = await coordinator.reconcile_once(
        binding_id="binding-1",
        restart=False,
        operation_id=operation_id,
        trigger_delivery_id=parent,
    )
    assert retry.delivery_id == run.final_delivery_id
    assert retry.checkpoint_advanced is True
    assert retry.has_more is False
    assert retry.changes_count == 0
    assert len(facade_ref.read_calls) == reads_before
    assert len(sink.calls) == sink_before
    if hasattr(state, "states"):
        assert len(state.states) == state_before
    assert len(checkpoint.commit_calls) == checkpoint_before
    assert runs.runs[("tenant-1", "binding-1")].record_version == version_before
