# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 — reconstruction quality (read-only execution history from durable evidence)."""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from testing_support.npsc5f_r4_regression_matrix import run_mandatory_regression_matrix

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.diagnostics.execution_reconstruction import (
    ExecutionReconstructionIntegrityError,
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
    attempt_id: AttemptId | None = None,
    timestamps: list[datetime] | None = None,
) -> None:
    resolved_attempt = attempt_id or mint_attempt_id()
    for index in range(count):
        event = sample_runtime_event(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=resolved_attempt,
        )
        if timestamps is not None:
            event = event.model_copy(update={"timestamp": timestamps[index]})
        store.append(event, tenant_id=tenant_id)


class _CorruptRuntimePersistence(InMemoryRuntimeEventStore):
    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through=None,
        after=None,
    ):
        rows = super().list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=through,
            after=after,
        )
        if not rows:
            return rows
        first = rows[0]
        corrupted_event = first.event.model_copy(update={"run_id": mint_run_id()})
        return [
            type(first)(event=corrupted_event, position=first.position),
            *rows[1:],
        ]


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


def test_npsc5f_r4_quality_read_only_never_calls_append(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    _append_events(store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, count=1)

    def _forbidden_append(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reconstruction must not append runtime events")

    monkeypatch.setattr(store, "append", _forbidden_append)
    view = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence()).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )
    assert view.is_runtime_history_complete


def test_npsc5f_r4_quality_determinism_same_evidence_same_view() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    _append_events(store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, count=3)
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    first = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    second = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    assert first == second


def test_npsc5f_r4_quality_corrupted_runtime_evidence_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    store = _CorruptRuntimePersistence()
    _append_events(store, tenant_id=_TENANT_A, task_id=task_id, run_id=run_id, count=1)
    with pytest.raises(ExecutionReconstructionIntegrityError, match="run_id"):
        ExecutionReconstructor(store, InMemoryCausalEvidencePersistence()).reconstruct_execution(
            _TENANT_A,
            task_id,
            run_id,
        )


def test_npsc5f_r4_quality_attempt_ordering_stable_across_attempts() -> None:
    store = InMemoryRuntimeEventStore()
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    _append_events(
        store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        count=1,
        attempt_id=attempt_b,
    )
    _append_events(
        store,
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        count=1,
        attempt_id=attempt_a,
    )
    view = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence()).reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
    )
    assert view.attempt_count == 2
    # Attempt projection follows run-local journal order (attempt_b first at position 1).
    assert [attempt.attempt_id for attempt in view.attempts] == [attempt_b, attempt_a]


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
    assert "FROZEN / PASS" in text
    assert "RuntimeEventPersistence" in text
    final_doc = (
        _REPO_ROOT
        / "docs/project/maintainers/qualification/NPSC_5F_R4_FINAL_RECONSTRUCTION_QUALITY_QUALIFICATION_AND_FREEZE.md"
    )
    assert final_doc.is_file()
    final_text = final_doc.read_text(encoding="utf-8")
    assert "FROZEN / PASS" in final_text
