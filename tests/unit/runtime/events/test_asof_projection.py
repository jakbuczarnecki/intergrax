# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.asof_projection import (
    HistoricalEventReference,
    InvalidRunExecutionHistoryError,
    RunExecutionBoundaryNotFoundError,
    RunExecutionHistoryNotFoundError,
    RunExecutionHistoryTruncatedError,
    RunExecutionLifecycleStatus,
    RunLifecycleViolationKind,
    apply_lifecycle_event,
    project_run_execution_as_of,
    reconstruct_run_execution_as_of,
)
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    PositionedRuntimeEvent,
    as_of_boundary_for_positioned,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import (
    PositionedJournalBoundaryNotFoundError,
    PositionedJournalPrefixTruncatedError,
    load_positioned_run_journal_through,
)

pytestmark = [pytest.mark.gate]

_TENANT = "tenant-a"
_SAME_TS = datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc)


def _event(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    event_type: RuntimeEventType,
    timestamp: datetime | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=mint_event_id(),
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_type=event_type,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=timestamp or _SAME_TS,
        correlation_id=run_id,
    )


def _append_sequence(
    store: InMemoryRuntimeEventStore,
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    event_types: list[RuntimeEventType],
) -> list[PositionedRuntimeEvent]:
    positioned: list[PositionedRuntimeEvent] = []
    for event_type in event_types:
        positioned.append(
            store.append(
                _event(
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    event_type=event_type,
                ),
                tenant_id=_TENANT,
            )
        )
    return positioned


@pytest.mark.unit
def test_prefix_projection_at_p4_excludes_later_events() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    sequence = [
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.PLAN_CREATED,
        RuntimeEventType.STEP_STARTED,
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_COMPLETED,
        RuntimeEventType.STEP_COMPLETED,
        RuntimeEventType.TASK_COMPLETED,
    ]
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=sequence,
    )
    boundary = as_of_boundary_for_positioned(positioned[3])
    prefix = tuple(store.list_positioned_through(boundary, tenant_id=_TENANT))
    projection = project_run_execution_as_of(
        boundary=boundary,
        positioned_events=prefix,
        tenant_id=_TENANT,
    )

    assert projection.boundary == boundary
    assert projection.last_included_position == positioned[3].position
    assert projection.lifecycle_status == RunExecutionLifecycleStatus.RUNNING
    assert projection.is_terminal is False
    assert len(projection.source_events) == 4
    assert projection.source_events[-1] == HistoricalEventReference(
        event_id=positioned[3].event_id,
        position=positioned[3].position,
        attempt_id=attempt_id,
        event_type=RuntimeEventType.TOOL_REQUESTED,
    )
    assert {ref.event_id for ref in projection.source_events}.isdisjoint(
        {positioned[4].event_id, positioned[5].event_id, positioned[6].event_id}
    )


@pytest.mark.unit
def test_terminal_projection_at_p7() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    sequence = [
        RuntimeEventType.TASK_CREATED,
        RuntimeEventType.PLAN_CREATED,
        RuntimeEventType.STEP_STARTED,
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_COMPLETED,
        RuntimeEventType.STEP_COMPLETED,
        RuntimeEventType.TASK_COMPLETED,
    ]
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=sequence,
    )
    boundary = as_of_boundary_for_positioned(positioned[6])
    projection = reconstruct_run_execution_as_of(
        persistence=store,
        tenant_id=_TENANT,
        boundary=boundary,
    )

    assert projection.lifecycle_status == RunExecutionLifecycleStatus.COMPLETED
    assert projection.is_terminal is True
    assert projection.last_included_position == positioned[6].position
    assert len(projection.source_events) == 7


@pytest.mark.unit
def test_same_timestamp_follows_execution_position() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    first = store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.TASK_CREATED,
            timestamp=_SAME_TS,
        ),
        tenant_id=_TENANT,
    )
    second = store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.PLAN_CREATED,
            timestamp=_SAME_TS,
        ),
        tenant_id=_TENANT,
    )
    boundary = as_of_boundary_for_positioned(second)
    projection = project_run_execution_as_of(
        boundary=boundary,
        positioned_events=tuple(store.list_positioned_through(boundary, tenant_id=_TENANT)),
        tenant_id=_TENANT,
    )
    assert [ref.event_id for ref in projection.source_events] == [
        first.event_id,
        second.event_id,
    ]


@pytest.mark.unit
def test_retry_attempt_history_and_current_attempt() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    positioned = [
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a1,
                event_type=RuntimeEventType.TASK_CREATED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a1,
                event_type=RuntimeEventType.TASK_FAILED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a1,
                event_type=RuntimeEventType.RETRY_SCHEDULED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a2,
                event_type=RuntimeEventType.RETRY_STARTED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_a2,
                event_type=RuntimeEventType.STEP_STARTED,
            ),
            tenant_id=_TENANT,
        ),
    ]
    at_failure = project_run_execution_as_of(
        boundary=as_of_boundary_for_positioned(positioned[1]),
        positioned_events=tuple(store.list_positioned_through(
            as_of_boundary_for_positioned(positioned[1]),
            tenant_id=_TENANT,
        )),
        tenant_id=_TENANT,
    )
    before_retry = project_run_execution_as_of(
        boundary=as_of_boundary_for_positioned(positioned[2]),
        positioned_events=tuple(store.list_positioned_through(
            as_of_boundary_for_positioned(positioned[2]),
            tenant_id=_TENANT,
        )),
        tenant_id=_TENANT,
    )
    after_retry = project_run_execution_as_of(
        boundary=as_of_boundary_for_positioned(positioned[-1]),
        positioned_events=tuple(store.list_positioned_through(
            as_of_boundary_for_positioned(positioned[-1]),
            tenant_id=_TENANT,
        )),
        tenant_id=_TENANT,
    )

    assert at_failure.lifecycle_status == RunExecutionLifecycleStatus.FAILED
    assert at_failure.is_terminal is False
    assert before_retry.lifecycle_status == RunExecutionLifecycleStatus.FAILED
    assert before_retry.is_terminal is False
    assert before_retry.current_attempt_id == attempt_a1
    assert before_retry.attempt_ids == (attempt_a1,)
    assert after_retry.lifecycle_status == RunExecutionLifecycleStatus.RUNNING
    assert after_retry.is_terminal is False
    assert after_retry.attempt_ids == (attempt_a1, attempt_a2)
    assert after_retry.current_attempt_id == attempt_a2
    assert len(after_retry.attempts) == 2


@pytest.mark.unit
def test_resume_preserves_attempt_identity() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = [
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=RuntimeEventType.TASK_CREATED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=RuntimeEventType.PAUSED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=RuntimeEventType.RESUMED,
            ),
            tenant_id=_TENANT,
        ),
    ]
    projection = project_run_execution_as_of(
        boundary=as_of_boundary_for_positioned(positioned[-1]),
        positioned_events=tuple(store.list_positioned_through(
            as_of_boundary_for_positioned(positioned[-1]),
            tenant_id=_TENANT,
        )),
        tenant_id=_TENANT,
    )
    assert projection.current_attempt_id == attempt_id
    assert projection.attempt_ids == (attempt_id,)
    assert projection.lifecycle_status == RunExecutionLifecycleStatus.RUNNING


@pytest.mark.unit
def test_cross_run_input_fails() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_a = mint_run_id()
    run_b = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned_a = store.append(
        _event(
            task_id=task_id,
            run_id=run_a,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.TASK_CREATED,
        ),
        tenant_id=_TENANT,
    )
    positioned_b = store.append(
        _event(
            task_id=task_id,
            run_id=run_b,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.TASK_CREATED,
        ),
        tenant_id=_TENANT,
    )
    boundary = as_of_boundary_for_positioned(positioned_a)
    with pytest.raises(InvalidRunExecutionHistoryError, match="run_id does not match"):
        project_run_execution_as_of(
            boundary=boundary,
            positioned_events=(positioned_a, positioned_b),
            tenant_id=_TENANT,
        )


@pytest.mark.unit
def test_out_of_order_and_duplicate_positions_fail() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    first = store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.TASK_CREATED,
        ),
        tenant_id=_TENANT,
    )
    second = store.append(
        _event(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=RuntimeEventType.PLAN_CREATED,
        ),
        tenant_id=_TENANT,
    )
    boundary = as_of_boundary_for_positioned(second)
    with pytest.raises(InvalidRunExecutionHistoryError, match="strict increasing"):
        project_run_execution_as_of(
            boundary=boundary,
            positioned_events=(second, first),
            tenant_id=_TENANT,
        )
    duplicate = PositionedRuntimeEvent(event=second.event, position=first.position)
    with pytest.raises(InvalidRunExecutionHistoryError, match="duplicate"):
        project_run_execution_as_of(
            boundary=boundary,
            positioned_events=(first, duplicate),
            tenant_id=_TENANT,
        )


@pytest.mark.unit
def test_truncated_prefix_fails_closed() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[RuntimeEventType.TASK_CREATED] * 5,
    )
    boundary = as_of_boundary_for_positioned(positioned[-1])
    with pytest.raises(PositionedJournalPrefixTruncatedError):
        load_positioned_run_journal_through(
            store,
            tenant_id=_TENANT,
            boundary=boundary,
            initial_limit=2,
            max_limit=4,
        )
    with pytest.raises(RunExecutionHistoryTruncatedError):
        reconstruct_run_execution_as_of(
            persistence=store,
            tenant_id=_TENANT,
            boundary=boundary,
            initial_limit=2,
            max_limit=4,
        )


@pytest.mark.unit
def test_unknown_history_fails_explicitly() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(1))
    with pytest.raises(RunExecutionHistoryNotFoundError):
        reconstruct_run_execution_as_of(
            persistence=store,
            tenant_id=_TENANT,
            boundary=boundary,
        )


@pytest.mark.unit
def test_nonexistent_boundary_position_rejected_by_reducer() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[RuntimeEventType.TASK_CREATED, RuntimeEventType.PLAN_CREATED],
    )
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(99))
    prefix = tuple(store.list_positioned_through(boundary, tenant_id=_TENANT))
    with pytest.raises(RunExecutionBoundaryNotFoundError):
        project_run_execution_as_of(
            boundary=boundary,
            positioned_events=prefix,
            tenant_id=_TENANT,
        )


@pytest.mark.unit
def test_nonexistent_boundary_position_rejected_by_reconstruction() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[RuntimeEventType.TASK_CREATED, RuntimeEventType.PLAN_CREATED],
    )
    boundary = AsOfBoundary(run_id=run_id, position=ExecutionEventPosition(99))
    with pytest.raises(RunExecutionBoundaryNotFoundError):
        reconstruct_run_execution_as_of(
            persistence=store,
            tenant_id=_TENANT,
            boundary=boundary,
        )
    with pytest.raises(PositionedJournalBoundaryNotFoundError):
        load_positioned_run_journal_through(
            store,
            tenant_id=_TENANT,
            boundary=boundary,
        )


@pytest.mark.unit
def test_valid_boundary_remains_stable_after_later_appends() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[RuntimeEventType.TASK_CREATED, RuntimeEventType.PLAN_CREATED],
    )
    boundary = as_of_boundary_for_positioned(positioned[1])
    projection1 = reconstruct_run_execution_as_of(
        persistence=store,
        tenant_id=_TENANT,
        boundary=boundary,
    )
    _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.STEP_STARTED,
            RuntimeEventType.STEP_COMPLETED,
        ],
    )
    projection2 = reconstruct_run_execution_as_of(
        persistence=store,
        tenant_id=_TENANT,
        boundary=boundary,
    )

    assert projection1 == projection2
    assert projection1.last_included_position == boundary.position


@pytest.mark.unit
def test_completed_run_cannot_reopen() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = [
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=RuntimeEventType.TASK_CREATED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=RuntimeEventType.TASK_COMPLETED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=mint_attempt_id(),
                event_type=RuntimeEventType.RETRY_STARTED,
            ),
            tenant_id=_TENANT,
        ),
    ]
    boundary = as_of_boundary_for_positioned(positioned[-1])
    with pytest.raises(InvalidRunExecutionHistoryError, match="after terminal completed"):
        project_run_execution_as_of(
            boundary=boundary,
            positioned_events=tuple(store.list_positioned_through(boundary, tenant_id=_TENANT)),
            tenant_id=_TENANT,
        )


@pytest.mark.unit
def test_cancelled_projection_is_terminal() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.CANCELLED,
        ],
    )
    boundary = as_of_boundary_for_positioned(positioned[-1])
    projection = project_run_execution_as_of(
        boundary=boundary,
        positioned_events=tuple(store.list_positioned_through(boundary, tenant_id=_TENANT)),
        tenant_id=_TENANT,
    )
    assert projection.lifecycle_status == RunExecutionLifecycleStatus.CANCELLED
    assert projection.is_terminal is True


@pytest.mark.unit
def test_cancelled_run_cannot_reopen() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = [
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=RuntimeEventType.TASK_CREATED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_type=RuntimeEventType.CANCELLED,
            ),
            tenant_id=_TENANT,
        ),
        store.append(
            _event(
                task_id=task_id,
                run_id=run_id,
                attempt_id=mint_attempt_id(),
                event_type=RuntimeEventType.RETRY_SCHEDULED,
            ),
            tenant_id=_TENANT,
        ),
    ]
    boundary = as_of_boundary_for_positioned(positioned[-1])
    with pytest.raises(InvalidRunExecutionHistoryError, match="after terminal cancelled"):
        project_run_execution_as_of(
            boundary=boundary,
            positioned_events=tuple(store.list_positioned_through(boundary, tenant_id=_TENANT)),
            tenant_id=_TENANT,
        )


@pytest.mark.unit
def test_apply_lifecycle_event_raises_typed_violation_fields() -> None:
    with pytest.raises(InvalidRunExecutionHistoryError) as exc_info:
        apply_lifecycle_event(
            RunExecutionLifecycleStatus.FAILED,
            RuntimeEventType.TASK_COMPLETED,
        )
    exc = exc_info.value
    assert exc.kind is RunLifecycleViolationKind.CONFLICTING_FINAL_OUTCOME
    assert exc.current_status is RunExecutionLifecycleStatus.FAILED
    assert exc.event_type is RuntimeEventType.TASK_COMPLETED

    with pytest.raises(InvalidRunExecutionHistoryError) as exc_info:
        apply_lifecycle_event(
            RunExecutionLifecycleStatus.COMPLETED,
            RuntimeEventType.TASK_CREATED,
        )
    exc = exc_info.value
    assert exc.kind is RunLifecycleViolationKind.EVENT_AFTER_TERMINAL
    assert exc.current_status is RunExecutionLifecycleStatus.COMPLETED
    assert exc.event_type is RuntimeEventType.TASK_CREATED


@pytest.mark.unit
def test_source_provenance_matches_positioned_events() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.PLAN_CREATED,
        ],
    )
    boundary = as_of_boundary_for_positioned(positioned[-1])
    prefix = tuple(store.list_positioned_through(boundary, tenant_id=_TENANT))
    projection = project_run_execution_as_of(
        boundary=boundary,
        positioned_events=prefix,
        tenant_id=_TENANT,
    )
    for ref, row in zip(projection.source_events, prefix, strict=True):
        assert ref.event_id == row.event_id
        assert ref.position == row.position


@pytest.mark.unit
def test_successful_projection_last_included_equals_boundary() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    positioned = _append_sequence(
        store,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        event_types=[
            RuntimeEventType.TASK_CREATED,
            RuntimeEventType.PLAN_CREATED,
            RuntimeEventType.STEP_STARTED,
        ],
    )
    boundary = as_of_boundary_for_positioned(positioned[1])
    projection = reconstruct_run_execution_as_of(
        persistence=store,
        tenant_id=_TENANT,
        boundary=boundary,
    )
    assert projection.last_included_position == boundary.position
