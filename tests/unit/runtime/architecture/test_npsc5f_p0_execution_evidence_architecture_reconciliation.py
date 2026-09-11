# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/P0 — execution evidence, replay & observability architecture reconciliation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from testing_support.npsc5f_r1_protected_drift import collect_r1_protected_production_drift
from testing_support.npsc5f_r2_protected_drift import R2_IMPLEMENTATION_SHA

from intergrax.contracts.execution_identity import RunId, mint_event_id, mint_run_id, mint_task_id
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.evidence_durability import (
    EvidencePersistenceRequirement,
    evidence_persistence_requirement,
)
from intergrax.runtime.events.persistence_contract import (
    EvidenceTenantRoutingMismatchError,
    MandatoryEvidencePersistenceError,
    RuntimeEventPersistence,
    RuntimeEventPersistenceIntegrityError,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.runtime.events.unified_run_journal import build_unified_run_journal
from intergrax.runtime.nexus.tracing.persistence_models import PersistedRun, RunMetadata, RunStats
from intergrax.runtime.observability.journal_export import serialize_runtime_event
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_EVIDENCE_ROOTS = (
    _REPO_ROOT / "intergrax" / "runtime" / "events",
    _REPO_ROOT / "intergrax" / "runtime" / "observability",
)

_FORBIDDEN_CONTROL_PLANE_SYMBOLS = frozenset(
    {
        "ChildExecutionRunner",
        "AttemptLifecycleService",
        "FanOutPartialRecoveryService",
        "LongRunningCoordinator",
        "transition_to_next_attempt",
        "schedule",
    },
)

_LINEAGE_MUTATION_CALLS = frozenset(
    {
        "open_segment",
        "admit_root",
        "admit_child",
        "seal_attempt",
        "mint_execution_id",
        "mint_child_execution_id",
    },
)


def _persisted_run(run_id: str, tenant_id: str) -> PersistedRun:
    return PersistedRun(
        metadata=RunMetadata(
            run_id=run_id,
            session_id="s1",
            user_id="u1",
            tenant_id=tenant_id,
            started_at_utc="2026-06-07T10:00:00+00:00",
            stats=RunStats(duration_ms=1, llm_usage={}),
        ),
        events=[],
    )


def _append_n_events(
    store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    run_id: RunId,
    count: int,
) -> None:
    task_id = mint_task_id()
    for _ in range(count):
        store.append(
            sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id),
            tenant_id=tenant_id,
        )


@pytest.mark.gate
def test_npsc5f_p0_same_event_id_different_payload_blocked() -> None:
    store = InMemoryRuntimeEventStore()
    tenant_id = "tenant-idempotency"
    event_id = mint_event_id()
    run_id = mint_run_id()
    task_id = mint_task_id()
    original = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
    )
    store.append(original, tenant_id=tenant_id)
    conflicting = sample_runtime_event(
        tenant_id=tenant_id,
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
    ).model_copy(update={"event_type": RuntimeEventType.STEP_COMPLETED})
    with pytest.raises(RuntimeEventPersistenceIntegrityError, match="conflicts"):
        store.append(conflicting, tenant_id=tenant_id)


@pytest.mark.gate
def test_npsc5f_p0_tenant_routing_explicit_vs_event_mismatch_blocked() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id="tenant-on-event")
    with pytest.raises(EvidenceTenantRoutingMismatchError, match="tenant"):
        store.append(event, tenant_id="tenant-on-route")
    assert store.get_by_event_id(tenant_id="tenant-on-event", event_id=event.event_id) is None
    assert store.get_by_event_id(tenant_id="tenant-on-route", event_id=event.event_id) is None


@pytest.mark.gate
def test_npsc5f_p0_build_unified_run_journal_silent_truncation_gap(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "journal_trunc.db")
    tenant_id = "tenant-trunc"
    run_id = mint_run_id()
    _append_n_events(store, tenant_id=tenant_id, run_id=run_id, count=5)
    journal = build_unified_run_journal(
        _persisted_run(run_id, tenant_id),
        runtime_store=store,
        page_size=3,
    )
    assert len(journal) == 5
    store.close()


@pytest.mark.gate
def test_npsc5f_p0_journal_export_uses_safe_export_envelope() -> None:
    from intergrax.runtime.observability.export_boundary import (
        FORBIDDEN_EXPORT_CONTENT_FIELDS,
        envelope_is_content_safe,
    )

    event = sample_runtime_event(tenant_id="tenant-export").model_copy(
        update={
            "payload": {
                "tool_id": "demo.tool",
                "latency_ms": 3,
                "prompt": "SECRET_PROMPT_123",
                "token": "SECRET_TOKEN_456",
            }
        }
    )
    envelope = serialize_runtime_event(event)
    assert envelope.tenant_id == "tenant-export"
    assert envelope.event_id == event.event_id
    assert envelope_is_content_safe(envelope)
    serialized = envelope.model_dump_json()
    for key in FORBIDDEN_EXPORT_CONTENT_FIELDS:
        assert f'"{key}"' not in serialized
    source = (_REPO_ROOT / "intergrax" / "runtime" / "observability" / "journal_export.py").read_text(
        encoding="utf-8",
    )
    assert "event.model_dump(mode=\"json\")" not in source


@pytest.mark.gate
def test_npsc5f_p0_execution_position_scoped_per_run(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "ordering.db")
    tenant_id = "tenant-order"
    run_a = mint_run_id()
    run_b = mint_run_id()
    task_id = mint_task_id()
    pos_a = store.append(
        sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_a),
        tenant_id=tenant_id,
    ).position.value
    pos_b = store.append(
        sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_b),
        tenant_id=tenant_id,
    ).position.value
    assert pos_a == pos_b == 1
    store.close()


@pytest.mark.gate
def test_npsc5f_p0_list_for_task_not_global_order_across_runs(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "task_order.db")
    tenant_id = "tenant-task"
    task_id = mint_task_id()
    run_a = mint_run_id()
    run_b = mint_run_id()
    store.append(
        sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_a),
        tenant_id=tenant_id,
    )
    store.append(
        sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_b),
        tenant_id=tenant_id,
    )
    by_task = store.list_for_task(task_id, tenant_id=tenant_id)
    assert len(by_task) == 2
    assert {evt.run_id for evt in by_task} == {run_a, run_b}
    store.close()


@pytest.mark.gate
def test_npsc5f_p0_concurrent_sqlite_positions_unique(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store = SQLiteRuntimeEventStore(db_path=tmp_path / "concurrent.db")
    tenant_id = "tenant-conc"
    run_id = mint_run_id()
    task_id = mint_task_id()

    def _one() -> int:
        positioned = store.append(
            sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id),
            tenant_id=tenant_id,
        )
        return positioned.position.value

    with ThreadPoolExecutor(max_workers=8) as pool:
        positions = list(pool.map(lambda _: _one(), range(16)))
    assert len(positions) == len(set(positions))
    assert sorted(positions) == list(range(1, 17))
    store.close()


@pytest.mark.gate
def test_npsc5f_p0_unknown_runtime_event_schema_rejected() -> None:
    inner = InMemoryRuntimeEventStore()
    store = ValidatingRuntimeEventPersistence(inner)
    event = sample_runtime_event(tenant_id="tenant-schema").model_copy(
        update={"schema_version": "runtime_event.v99"},
    )
    with pytest.raises(Exception):
        store.append(event, tenant_id="tenant-schema")
    store.close()


@pytest.mark.gate
def test_npsc5f_p0_cross_tenant_read_returns_none() -> None:
    store = InMemoryRuntimeEventStore()
    event = sample_runtime_event(tenant_id="tenant-a")
    store.append(event, tenant_id="tenant-a")
    assert store.get_by_event_id(tenant_id="tenant-b", event_id=event.event_id) is None
    assert store.list_for_run(event.run_id, tenant_id="tenant-b") == []


@pytest.mark.gate
def test_npsc5f_p0_evidence_layer_does_not_mutate_lineage() -> None:
    violations: list[str] = []
    for root in _EVIDENCE_ROOTS:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in _LINEAGE_MUTATION_CALLS:
                        rel = path.relative_to(_REPO_ROOT).as_posix()
                        violations.append(f"{rel}:{node.lineno}:{node.func.attr}")
    assert violations == []


def _call_symbol(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


@pytest.mark.gate
def test_npsc5f_p0_no_execution_control_bypass_from_evidence_roots() -> None:
    violations: list[str] = []
    for root in _EVIDENCE_ROOTS:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                symbol = _call_symbol(node.func)
                if symbol in _FORBIDDEN_CONTROL_PLANE_SYMBOLS:
                    rel = path.relative_to(_REPO_ROOT).as_posix()
                    violations.append(f"{rel}:{node.lineno}:{symbol}")
    assert violations == []


@pytest.mark.gate
def test_npsc5f_p0_sqlite_evidence_readable_from_second_store_instance(tmp_path: Path) -> None:
    db_path = tmp_path / "cross_instance.db"
    tenant_id = "tenant-xproc"
    run_id = mint_run_id()
    writer = SQLiteRuntimeEventStore(db_path=db_path)
    event = sample_runtime_event(tenant_id=tenant_id, run_id=run_id)
    writer.append(event, tenant_id=tenant_id)
    writer.close()
    reader = SQLiteRuntimeEventStore(db_path=db_path)
    rows = reader.list_for_run(run_id, tenant_id=tenant_id)
    reader.close()
    assert len(rows) == 1
    assert rows[0].event_id == event.event_id


@pytest.mark.gate
def test_npsc5f_p0_bus_mandatory_persistence_failure_is_fail_closed() -> None:
    class _Failing(RuntimeEventPersistence):
        def append(self, event, *, tenant_id: str):
            raise RuntimeError("sink down")

        def list_positioned_for_run(
            self,
            run_id,
            *,
            tenant_id: str,
            limit: int = 1000,
            through=None,
            after=None,
        ):
            return []

        def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
            return []

        def list_positioned_for_task_grouped_by_run(self, task_id, *, tenant_id: str, limit: int = 1000):
            from intergrax.runtime.events.persistence_contract import TaskRuntimeEventRuns

            return TaskRuntimeEventRuns(runs=())

        def get_by_event_id(self, *, tenant_id: str, event_id):
            return None

    bus = RuntimeEventBus(persistence=_Failing(), record_history=True)
    event = sample_runtime_event(tenant_id="tenant-bus")
    with pytest.raises(MandatoryEvidencePersistenceError, match="mandatory"):
        bus.record(event, tenant_id="tenant-bus")
    assert bus.history == []


@pytest.mark.gate
def test_npsc5f_p0_bus_best_effort_persistence_failure_allows_record() -> None:
    class _Failing(RuntimeEventPersistence):
        def append(self, event, *, tenant_id: str):
            raise RuntimeError("sink down")

        def list_positioned_for_run(
            self,
            run_id,
            *,
            tenant_id: str,
            limit: int = 1000,
            through=None,
            after=None,
        ):
            return []

        def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
            return []

        def list_positioned_for_task_grouped_by_run(self, task_id, *, tenant_id: str, limit: int = 1000):
            from intergrax.runtime.events.persistence_contract import TaskRuntimeEventRuns

            return TaskRuntimeEventRuns(runs=())

        def get_by_event_id(self, *, tenant_id: str, event_id):
            return None

    bus = RuntimeEventBus(persistence=_Failing(), record_history=True)
    event = sample_runtime_event(tenant_id="tenant-bus").model_copy(
        update={"event_type": RuntimeEventType.TASK_PROGRESS, "phase": ExecutionPhase.STEP_EXECUTION},
    )
    from intergrax.runtime.events.event_catalog import should_persist_event

    event_id = mint_event_id()
    while not should_persist_event(event.model_copy(update={"event_id": event_id})):
        event_id = mint_event_id()
    event = event.model_copy(update={"event_id": event_id})
    assert (
        evidence_persistence_requirement(event)
        is EvidencePersistenceRequirement.BEST_EFFORT
    )
    bus.record(event, tenant_id="tenant-bus")
    assert bus.history[-1].event_id == event.event_id


@pytest.mark.gate
def test_npsc5f_p0_unified_journal_exposes_explicit_completeness_contract() -> None:
    source = (
        _REPO_ROOT / "intergrax" / "runtime" / "events" / "unified_run_journal.py"
    ).read_text(encoding="utf-8")
    assert "class RunJournalReadPage" in source
    assert "is_complete" in source
    assert "next_cursor" in source
    assert "read_run_journal_page" in source
    assert "load_complete_run_journal" in source


@pytest.mark.gate
def test_npsc5f_p0_r1_protected_evidence_surfaces_have_no_unqualified_post_r1_drift() -> None:
    # R2 journal read contracts qualified at R2_IMPLEMENTATION_SHA; sentinel guards post-R2 only.
    drift = collect_r1_protected_production_drift(
        _REPO_ROOT,
        from_sha=R2_IMPLEMENTATION_SHA,
    )
    assert drift == [], f"unexpected R1 protected production drift since R2 freeze: {drift}"


# P0 qualification flags (inventory — gaps do not fail P0)
SAME_EVENT_ID_DIFFERENT_PAYLOAD = "BLOCKED"
TENANT_ROUTING_EXPLICIT_VS_EVENT = "BLOCKED"
FULL_RUN_READ = "FIXED_R2"
STREAM_COMPLETENESS_EXPLICIT = "YES"
RAW_EXPORT_BYPASS = "GAP"
EVIDENCE_PERSISTENCE_FAILURE = "FAIL_CLOSED_MANDATORY"
CROSS_PROCESS_EVIDENCE = "PASS"
CONCURRENT_POSITION_ALLOCATION = "PASS"
DIRECT_EXECUTION_BYPASSES_FROM_EVIDENCE = 0
SECOND_LINEAGE_OWNER = False
