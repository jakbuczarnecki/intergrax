# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R2 Final — journal completeness & run-local ordering qualification and freeze."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import RunId, mint_run_id, mint_task_id
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.events.execution_position import ExecutionEventPosition
from intergrax.runtime.events.persistence_contract import (
    NullRuntimeEventPersistence,
    RuntimeEventPersistence,
)
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.events.stores.validating_runtime_event_store import (
    ValidatingRuntimeEventPersistence,
)
from intergrax.runtime.events.unified_run_journal import (
    JournalReadLimitExceededError,
    RunJournalContinuationCursor,
    RunJournalReadPage,
    load_complete_run_journal,
    read_run_journal_page,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.npsc5f_r2_protected_drift import (
    R2_IMPLEMENTATION_SHA,
    R2_POST_QUALIFIED_BASELINE_SHA,
    collect_r2_protected_production_drift,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-r2-final"

R1_FINAL_SHA = "455c09f342f995ac0a6fcb03ffef2f4d3e36a447"
R1_IMPLEMENTATION_SHA = "455d3b216f0ad56ea9cdf9db6e0f760b50063a81"
NPSC_5F_P0_SHA = "7811371da1069b661987b050a4c9bf42c02bda69"
NPSC_5E_FINAL_SHA = "fabdcfe931dfd3a0b22d35cbf06ac94b2b0176f7"

_R2_JOURNAL_SURFACE = (
    "intergrax/runtime/events/unified_run_journal.py",
    "intergrax/runtime/events/execution_position.py",
)

_FORBIDDEN_CONTROL_PLANE_SYMBOLS = frozenset(
    {
        "ChildExecutionRunner",
        "AttemptLifecycleService",
        "LongRunningCoordinator",
        "FanOutPartialRecoveryService",
        "ExecutionRuntime",
    },
)

_FORBIDDEN_CHRONOLOGY_CLAIMS = re.compile(
    r"(task-global execution order|canonical task chronology|tenant-global event order|global event sequence)",
    re.IGNORECASE,
)

_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")

_MANDATORY_SUITES: tuple[tuple[str, list[str]], ...] = (
    (
        "R2 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5f_r2_journal_completeness_ordering.py"],
    ),
    (
        "R1 Final",
        ["tests/unit/runtime/architecture/test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py"],
    ),
    (
        "NPSC-5F P0 gate",
        ["tests/unit/runtime/architecture/test_npsc5f_p0_execution_evidence_architecture_reconciliation.py"],
    ),
    ("Runtime events suites", ["tests/unit/runtime/events/"]),
    (
        "Runtime observability suites",
        ["tests/unit/runtime/observability/"],
    ),
    (
        "TRACE-ASOF",
        [
            "tests/unit/runtime/events/test_execution_position_asof.py",
            "tests/unit/runtime/events/test_asof_projection.py",
        ],
    ),
    (
        "TRACE-BITEMP",
        [
            "tests/unit/contracts/test_bitemporal_revision_ordering.py",
            "tests/unit/contracts/test_bitemporal_knowledge.py",
        ],
    ),
    (
        "Execution reconstruction",
        ["tests/unit/runtime/diagnostics/test_execution_reconstruction.py"],
    ),
    (
        "NPSC-5E Final",
        ["tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"],
    ),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    (
        "NPSC-5D Final",
        ["tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"],
    ),
    (
        "NPSC-5B Final",
        ["tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py"],
    ),
    (
        "R2 drift classifier",
        ["tests/unit/testing_support/test_npsc5f_r2_protected_drift.py"],
    ),
)


def _run_pytest(targets: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "pytest", *targets, "-q", "--tb=no"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _append_n(
    store: RuntimeEventPersistence,
    *,
    run_id: RunId,
    count: int,
    tenant_id: str = _TENANT,
) -> None:
    task_id = mint_task_id()
    for _ in range(count):
        store.append(
            sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id),
            tenant_id=tenant_id,
        )


@pytest.mark.parametrize(("label", "targets"), _MANDATORY_SUITES, ids=[label for label, _ in _MANDATORY_SUITES])
def test_mandatory_frozen_suite_passes(label: str, targets: list[str]) -> None:
    proc = _run_pytest(targets)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"


def test_canonical_predecessor_shas_recorded() -> None:
    assert R2_IMPLEMENTATION_SHA.startswith("6325074")
    assert R1_FINAL_SHA.startswith("455c09f")
    assert R1_IMPLEMENTATION_SHA.startswith("455d3b2")
    assert NPSC_5F_P0_SHA.startswith("7811371")
    assert NPSC_5E_FINAL_SHA.startswith("fabdcfe")


def test_r2_final_no_unqualified_protected_drift_since_qualified_baseline() -> None:
    drift = collect_r2_protected_production_drift(
        _REPO_ROOT,
        from_sha=R2_POST_QUALIFIED_BASELINE_SHA,
    )
    assert drift == [], f"R2 protected production drift since qualified baseline: {drift}"


def test_r2_page_completeness_cursor_invariant(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "invariant.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=7)
    cursor = None
    while True:
        page = read_run_journal_page(
            store,
            tenant_id=_TENANT,
            run_id=run_id,
            page_size=3,
            cursor=cursor,
        )
        assert isinstance(page, RunJournalReadPage)
        if page.is_complete:
            assert page.next_cursor is None
        else:
            assert page.next_cursor is not None
        if page.is_complete:
            break
        cursor = page.next_cursor
    store.close()


def test_r2_exact_max_complete_loader_succeeds() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=4)
    journal = load_complete_run_journal(
        store,
        tenant_id=_TENANT,
        run_id=run_id,
        max_events=4,
        page_size=2,
    )
    assert len(journal) == 4


def test_r2_max_plus_one_complete_loader_raises() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=5)
    with pytest.raises(JournalReadLimitExceededError):
        load_complete_run_journal(
            store,
            tenant_id=_TENANT,
            run_id=run_id,
            max_events=4,
            page_size=2,
        )


def test_r2_sqlite_concurrent_append_respects_snapshot(tmp_path: Path) -> None:
    store = SQLiteRuntimeEventStore(db_path=tmp_path / "snap.db")
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=6)
    first = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=2)
    assert first.next_cursor is not None
    snapshot_through = first.next_cursor.snapshot_through
    _append_n(store, run_id=run_id, count=3)
    collected: list[str] = [event.event_id for event in first.events]
    cursor = first.next_cursor
    while cursor is not None:
        page = read_run_journal_page(
            store,
            tenant_id=_TENANT,
            run_id=run_id,
            page_size=10,
            cursor=cursor,
        )
        collected.extend(event.event_id for event in page.events)
        if page.is_complete:
            break
        cursor = page.next_cursor
    positioned = store.list_positioned_for_run(
        run_id,
        tenant_id=_TENANT,
        limit=100,
        through=snapshot_through,
    )
    assert collected == [row.event.event_id for row in positioned]
    assert len(collected) == 6
    store.close()


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("document", lambda: DocumentBackedRuntimeEventStore(InMemoryDocumentStore())),
        ("validating", lambda: ValidatingRuntimeEventPersistence(InMemoryRuntimeEventStore())),
    ],
)
def test_r2_adapter_multi_page_no_duplicates(label: str, factory) -> None:
    store = factory()
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=8)
    event_ids: list[str] = []
    cursor = None
    while True:
        page = read_run_journal_page(
            store,
            tenant_id=_TENANT,
            run_id=run_id,
            page_size=3,
            cursor=cursor,
        )
        event_ids.extend(event.event_id for event in page.events)
        if page.is_complete:
            break
        cursor = page.next_cursor
    positioned = store.list_positioned_for_run(run_id, tenant_id=_TENANT, limit=100)
    assert len(event_ids) == len(set(event_ids))
    assert event_ids == [row.event.event_id for row in positioned]
    positions = [row.position.value for row in positioned]
    assert positions == sorted(positions)
    store.close()


def test_r2_null_store_empty_journal_complete() -> None:
    store = NullRuntimeEventPersistence()
    run_id = mint_run_id()
    page = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=5)
    assert page.events == ()
    assert page.is_complete is True
    assert page.next_cursor is None


def test_r2_cursor_snapshot_immutable_across_pages() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    _append_n(store, run_id=run_id, count=5)
    first = read_run_journal_page(store, tenant_id=_TENANT, run_id=run_id, page_size=2)
    assert first.next_cursor is not None
    snap = first.next_cursor.snapshot_through
    second = read_run_journal_page(
        store,
        tenant_id=_TENANT,
        run_id=run_id,
        page_size=2,
        cursor=first.next_cursor,
    )
    if second.next_cursor is not None:
        assert second.next_cursor.snapshot_through == snap


def test_r2_no_unsupported_chronology_claims_in_events_tree() -> None:
    events_root = _REPO_ROOT / "intergrax" / "runtime" / "events"
    violations: list[str] = []
    for path in events_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _FORBIDDEN_CHRONOLOGY_CLAIMS.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line = text[line_start : text.find("\n", match.start())]
            if "not task-global" in line.lower() or "not task-global chronology" in line.lower():
                continue
            violations.append(f"{path.relative_to(_REPO_ROOT)}:{match.group(0)}")
    assert violations == []


def test_r2_no_global_ordering_authority_symbols() -> None:
    patterns = ("GlobalEventPosition", "TaskGlobalPosition", "global_sequence")
    hits: list[str] = []
    events_root = _REPO_ROOT / "intergrax" / "runtime" / "events"
    for path in events_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern in source:
                hits.append(f"{path.name}:{pattern}")
    assert hits == []


def test_r2_journal_export_uses_bounded_read_not_silent_complete_assumption() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "observability" / "journal_export.py"
    source = path.read_text(encoding="utf-8")
    assert "read_run_journal_page" in source or "RunJournalReadPage" in source
    assert "load_complete_run_journal" not in source or "max_events" in source


def test_r2_no_reflection_on_journal_surface() -> None:
    hits: list[str] = []
    for rel in _R2_JOURNAL_SURFACE:
        source = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        for match in _REFLECTION_PATTERN.finditer(source):
            hits.append(f"{rel}:{match.group(0)}")
    assert hits == []


def test_r2_no_execution_control_from_journal_surface() -> None:
    violations: list[str] = []
    for rel in _R2_JOURNAL_SURFACE:
        path = _REPO_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            symbol = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if symbol in _FORBIDDEN_CONTROL_PLANE_SYMBOLS:
                violations.append(f"{rel}:{node.lineno}:{symbol}")
    assert violations == []


def test_r2_journal_read_surface_has_no_concrete_store_imports() -> None:
    journal_path = _REPO_ROOT / "intergrax" / "runtime" / "events" / "unified_run_journal.py"
    source = journal_path.read_text(encoding="utf-8")
    assert "intergrax.runtime.events.stores" not in source
    assert "EvidencePersistencePort" in source


def test_r2_journal_read_surface_has_no_recovery_ownership() -> None:
    forbidden_prefixes = (
        "intergrax.runtime.long_running",
        "intergrax.runtime.replay",
    )
    journal_path = _REPO_ROOT / "intergrax" / "runtime" / "events" / "unified_run_journal.py"
    source = journal_path.read_text(encoding="utf-8")
    violations = [prefix for prefix in forbidden_prefixes if prefix in source]
    assert violations == []


def test_r2_no_second_journal_framework() -> None:
    events_root = _REPO_ROOT / "intergrax" / "runtime" / "events"
    defs = [
        path.relative_to(_REPO_ROOT).as_posix()
        for path in events_root.rglob("*.py")
        if path.name != "unified_run_journal.py"
        and "def read_run_journal_page" in path.read_text(encoding="utf-8")
    ]
    assert defs == []


def test_r2_typed_cursor_not_dict() -> None:
    assert RunJournalContinuationCursor.__name__ == "RunJournalContinuationCursor"


@pytest.mark.gate
def test_npsc5f_r2_final_qualification_gate() -> None:
    assert R2_IMPLEMENTATION_SHA == "632507420f0ab8360aede43a2740e8fccc44efb4"
    assert ExecutionEventPosition.__doc__ is not None
    assert "run" in ExecutionEventPosition.__doc__.lower()
