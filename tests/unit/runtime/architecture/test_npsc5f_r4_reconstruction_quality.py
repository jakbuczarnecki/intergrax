# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 — reconstruction quality (read-only execution history from durable evidence)."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from testing_support.npsc5f_r4_regression_matrix import run_mandatory_regression_matrix

from intergrax.contracts.execution_identity import RunId, TaskId, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstructor,
    RuntimeHistoryCompleteness,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT_A = "tenant-r4-quality-a"
_TENANT_B = "tenant-r4-quality-b"

_RECONSTRUCTION_MODULE = _REPO_ROOT / "intergrax/runtime/diagnostics/execution_reconstruction.py"

_FORBIDDEN_CONTROL_PLANE = frozenset(
    {
        "ExecutionRuntime",
        "ChildExecutionRunner",
        "AttemptLifecycleService",
        "LongRunningCoordinator",
        "FanOutPartialRecoveryService",
        "ReplayRuntime",
        "HistoricalRuntime",
    },
)

_R1_R2_R3_IMPLEMENTATION_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "R1 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r1_durable_evidence_commit_tenant_integrity.py"],
    ),
    (
        "R2 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r2_journal_completeness_ordering.py"],
    ),
    (
        "R3 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r3_governed_evidence_export.py"],
    ),
)


def _append_events(
    store: InMemoryRuntimeEventStore,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    count: int,
    timestamps: list[datetime] | None = None,
) -> None:
    for index in range(count):
        event = sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id)
        if timestamps is not None:
            event = event.model_copy(update={"timestamp": timestamps[index]})
        store.append(event, tenant_id=tenant_id)


def test_npsc5f_r4_quality_static_no_execution_control_surface() -> None:
    tree = ast.parse(_RECONSTRUCTION_MODULE.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert names.isdisjoint(_FORBIDDEN_CONTROL_PLANE)


def test_npsc5f_r4_quality_complete_reconstruction() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    _append_events(store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, count=3)
    view = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence()).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )
    assert view.is_runtime_history_complete
    assert [row.position.value for row in view.positioned_events] == [1, 2, 3]
    assert view.attempt_count >= 1


def test_npsc5f_r4_quality_missing_evidence_explicit_incomplete() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    _append_events(store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, count=5)
    view = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence()).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
        initial_limit=2,
        max_limit=2,
    )
    assert view.runtime_history_completeness is RuntimeHistoryCompleteness.TRUNCATED
    assert len(view.positioned_events) == 2
    assert view.is_runtime_history_complete is False


def test_npsc5f_r4_quality_ordering_matches_journal_not_timestamp() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    base = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(hours=2), base, base + timedelta(hours=1)]
    _append_events(
        store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        count=3,
        timestamps=timestamps,
    )
    view = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence()).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )
    assert [row.position.value for row in view.positioned_events] == [1, 2, 3]
    assert [row.event.timestamp for row in view.positioned_events] == timestamps


def test_npsc5f_r4_quality_read_only_does_not_mutate_persistence() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    _append_events(store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, count=2)
    before = len(store._accepted_by_event_id)
    ExecutionReconstructor(store, InMemoryCausalEvidencePersistence()).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )
    assert len(store._accepted_by_event_id) == before


def test_npsc5f_r4_quality_tenant_isolation() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    _append_events(store, tenant_id=_TENANT_B, task_id=task_id, run_id=run_id, count=1)
    view = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence()).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )
    assert view.positioned_events == ()
    assert view.runtime_history_completeness is RuntimeHistoryCompleteness.COMPLETE


@pytest.mark.gate
def test_npsc5f_r4_quality_r1_r2_r3_regression() -> None:
    proc = run_mandatory_regression_matrix(_REPO_ROOT, suites=_R1_R2_R3_IMPLEMENTATION_SUITES)
    assert proc.returncode == 0, (
        f"R1/R2/R3 implementation regression failed:\n{proc.stdout}\n{proc.stderr}"
    )


def test_npsc5f_r4_quality_qualification_gate() -> None:
    doc = _REPO_ROOT / "docs/project/maintainers/qualification/NPSC_5F_R4_RECONSTRUCTION_QUALITY.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert "IMPLEMENTATION COMPLETE" in text
    assert "RuntimeEventPersistence" in text
