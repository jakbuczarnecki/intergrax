# © Artur Czarnecki. All rights reserved.

"""Contract ownership and semantics for ``TaskRuntimeEventRuns``."""

from __future__ import annotations

import dataclasses

import pytest

from intergrax.contracts.execution_event_position import ExecutionEventPosition
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.contracts.task_runtime_event_runs import TaskRuntimeEventRuns
from intergrax.runtime.events.persistence_contract import (
    TaskRuntimeEventRuns as LegacyTaskRuntimeEventRuns,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _positioned(run_id: str, position: int) -> PositionedRuntimeEvent:
    event = sample_runtime_event(run_id=run_id)
    return PositionedRuntimeEvent(
        event=event,
        position=ExecutionEventPosition(value=position),
    )


def test_legacy_runtime_import_path_is_canonical_contract_type() -> None:
    assert LegacyTaskRuntimeEventRuns is TaskRuntimeEventRuns


def test_task_runtime_event_runs_empty() -> None:
    dto = TaskRuntimeEventRuns(runs=())
    assert dto.runs == ()


def test_task_runtime_event_runs_single_run_multiple_events() -> None:
    run_id = mint_run_id()
    rows = (_positioned(run_id, 1), _positioned(run_id, 2))
    dto = TaskRuntimeEventRuns(runs=((run_id, rows),))
    assert len(dto.runs) == 1
    assert dto.runs[0][0] == run_id
    assert tuple(dto.runs[0][1]) == rows


def test_task_runtime_event_runs_multiple_runs_preserves_order() -> None:
    run_a = mint_run_id()
    run_b = mint_run_id()
    runs = (
        (run_a, (_positioned(run_a, 1),)),
        (run_b, (_positioned(run_b, 1), _positioned(run_b, 2))),
    )
    dto = TaskRuntimeEventRuns(runs=runs)
    assert dto.runs == runs


def test_task_runtime_event_runs_is_immutable() -> None:
    dto = TaskRuntimeEventRuns(runs=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        dto.runs = ()  # type: ignore[misc]
