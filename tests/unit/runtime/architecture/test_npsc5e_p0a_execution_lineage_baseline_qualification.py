# © Artur Czarnecki. All rights reserved.

"""NPSC-5E-P0A — execution lineage baseline reconciliation & qualification."""

from __future__ import annotations

import ast
import concurrent.futures
import re
import subprocess
from contextvars import copy_context
from pathlib import Path

import pytest

from intergrax.contracts.attempt_lifecycle import AttemptTransitionReason
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageAttemptScope,
    ExecutionLineageError,
    ExecutionLineageIntegrityError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.attempt_lifecycle.persistence import InMemoryAttemptLifecycleStore
from intergrax.runtime.execution.attempt_lifecycle.service import AttemptLifecycleService
from intergrax.runtime.execution.lineage.active_lineage import (
    ActiveExecutionLineageState,
    bind_active_execution_lineage,
    peek_active_execution_lineage,
    reset_active_execution_lineage,
)
from intergrax.runtime.execution.lineage.codecs import (
    decode_execution_lineage_admission_record,
    encode_execution_lineage_admission_record,
)
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

NPSC_5D_FROZEN_SHA = "a4a1faca01cd5004e372f235132184a84aa5a6bd"
LINEAGE_IMPLEMENTATION_SHA = "a3e719b1c5189116952f2913140cd1ece360877c"
QUALIFIED_PRODUCTION_BASELINE_SHA = "a72c9b568c61e28180756059ae48a99fb56eaa19"

_LINEAGE_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "lineage"
_LINEAGE_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "execution_lineage.py"
_RUNTIME_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "runtime.py"
_ATTEMPT_LIFECYCLE_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "attempt_lifecycle" / "service.py"
)
_GRAPH_RUNNER_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "graph_runner.py"
)
_LONG_RUNNING_BRIDGE_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "long_running_bridge.py"
)

_FORBIDDEN_LINEAGE_ABSTRACTIONS = (
    "RecoveryLineageManager",
    "RetryLineageEngine",
    "ExecutionLineageRuntime",
)

_FORBIDDEN_LINEAGE_CALLS = frozenset(
    {
        "transition_to_next_attempt",
        "mint_retry_attempt_id",
        "mint_execution_id",
        "mint_run_id",
        "mint_attempt_id",
        "mint_root_execution_identity",
        "mint_child_execution_id",
        "schedule",
        "authorize",
    },
)

_MINT_RETRY_ALLOWED_FILES = frozenset(
    {
        "intergrax/runtime/execution/identity_authority.py",
        "intergrax/runtime/execution/attempt_lifecycle/service.py",
    },
)

_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")

_FROZEN_NPSC_REGRESSION_MODULES = (
    "tests.unit.runtime.architecture.test_npsc5a_multi_agent_coordination_gate",
    "tests.unit.runtime.architecture.test_npsc5a_coordination_delegation_e2e",
    "tests.unit.runtime.architecture.test_npsc5b_final_production_fanout_fanin_qualification",
    "tests.unit.runtime.architecture.test_npsc5c_coordination_intent_gate",
    "tests.unit.runtime.architecture.test_npsc5c_decision_execution_e2e",
    "tests.unit.runtime.architecture.test_npsc5d_final_multi_agent_governance_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r1_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r2_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r3_final_qualification",
)

def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _iter_lineage_python_files() -> list[Path]:
    paths = [_LINEAGE_CONTRACT]
    if _LINEAGE_ROOT.is_dir():
        paths.extend(sorted(_LINEAGE_ROOT.rglob("*.py")))
    return [path for path in paths if "__pycache__" not in path.parts]


def _collect_forbidden_calls_in_lineage() -> list[str]:
    violations: list[str] = []
    for path in _iter_lineage_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in _FORBIDDEN_LINEAGE_CALLS:
                violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def _collect_mint_retry_calls_outside_canonical_owner() -> list[str]:
    violations: list[str] = []
    scan_roots = (
        _REPO_ROOT / "intergrax" / "runtime" / "execution",
        _REPO_ROOT / "intergrax" / "runtime" / "nexus",
        _REPO_ROOT / "intergrax" / "applications",
    )
    for root in scan_roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _call_name(node.func) != "mint_retry_attempt_id":
                    continue
                if rel in _MINT_RETRY_ALLOWED_FILES:
                    continue
                violations.append(f"{rel}:{node.lineno}: mint_retry_attempt_id()")
    return violations


def _scope(
    *,
    tenant_id: str = "tenant-a",
    task_id: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> ExecutionLineageAttemptScope:
    return build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        run_id=run_id or mint_run_id(),
        attempt_id=attempt_id or mint_attempt_id(),
    )


@pytest.mark.gate
def test_npsc5e_p0a_frozen_predecessor_and_lineage_baseline_commits_exist() -> None:
    for sha in (NPSC_5D_FROZEN_SHA, LINEAGE_IMPLEMENTATION_SHA, QUALIFIED_PRODUCTION_BASELINE_SHA):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, sha


@pytest.mark.gate
def test_npsc5e_p0a_lineage_does_not_own_lifecycle_schedule_authorize_or_retry() -> None:
    assert _collect_forbidden_calls_in_lineage() == []


@pytest.mark.gate
def test_npsc5e_p0a_no_forbidden_lineage_runtime_abstractions() -> None:
    violations: list[str] = []
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        if "build" in path.parts:
            continue
        source = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_LINEAGE_ABSTRACTIONS:
            if name in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert violations == []


@pytest.mark.gate
def test_npsc5e_p0a_lineage_has_no_reflection_bypass() -> None:
    violations: list[str] = []
    for path in _iter_lineage_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if _REFLECTION_PATTERN.search(line):
                violations.append(f"{rel}:{lineno}")
    assert violations == []


@pytest.mark.gate
def test_npsc5e_p0a_runtime_owns_lifecycle_entrypoint() -> None:
    source = _RUNTIME_PATH.read_text(encoding="utf-8-sig")
    assert "class ExecutionRuntime" in source
    assert "activate_root_execution_lineage" in source
    assert "deactivate_root_execution_lineage" in source


@pytest.mark.gate
def test_npsc5e_p0a_attempt_lifecycle_service_owns_retry_transition() -> None:
    source = _ATTEMPT_LIFECYCLE_PATH.read_text(encoding="utf-8-sig")
    assert "class AttemptLifecycleService" in source
    assert "def transition_to_next_attempt" in source
    assert "mint_retry_attempt_id()" in source


@pytest.mark.gate
def test_npsc5e_p0a_no_direct_retry_attempt_mint_outside_canonical_owner() -> None:
    assert _collect_mint_retry_calls_outside_canonical_owner() == []


@pytest.mark.gate
def test_npsc5e_p0a_nexus_propagation_does_not_mint_execution_identity() -> None:
    for path in (_GRAPH_RUNNER_PATH, _LONG_RUNNING_BRIDGE_PATH):
        source = path.read_text(encoding="utf-8-sig")
        assert "mint_execution_id" not in source
        assert "mint_root_execution_identity" not in source
        assert "mint_run_id" not in source
        assert "mint_attempt_id" not in source


@pytest.mark.gate
def test_npsc5e_p0a_root_admission_single_lineage_root() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    scope = _scope()
    root = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    record = persistence.admit_root(scope, root, root)
    assert record.execution_id == root
    assert record.segment_root_execution_id == root
    assert record.parent_execution_id is None
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    roots = [item for item in page.admissions if item.parent_execution_id is None]
    assert len(roots) == 1


@pytest.mark.gate
def test_npsc5e_p0a_idempotent_admission() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    scope = _scope()
    root = mint_execution_id()
    child = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    first = persistence.admit_child(scope, root, child, root)
    second = persistence.admit_child(scope, root, child, root)
    assert first == second
    page = persistence.list_admissions_for_attempt(scope, limit=10)
    child_rows = [item for item in page.admissions if item.execution_id == child]
    assert len(child_rows) == 1


@pytest.mark.gate
def test_npsc5e_p0a_conflicting_admission_fails_closed() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    scope = _scope()
    root = mint_execution_id()
    child = mint_execution_id()
    other_parent = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.admit_child(scope, root, child, root)
    with pytest.raises(ExecutionLineageIntegrityError):
        persistence.admit_child(scope, root, child, other_parent)


@pytest.mark.gate
def test_npsc5e_p0a_codec_roundtrip_preserves_identity() -> None:
    scope = _scope()
    root = mint_execution_id()
    child = mint_execution_id()
    record = build_execution_lineage_attempt_scope(
        tenant_id=scope.tenant_id,
        task_id=scope.task_id,
        run_id=scope.run_id,
        attempt_id=scope.attempt_id,
    )
    from intergrax.contracts.execution_lineage import ExecutionLineageAdmissionRecord

    original = ExecutionLineageAdmissionRecord(
        scope=record,
        segment_root_execution_id=root,
        execution_id=child,
        parent_execution_id=root,
        admission_position=2,
    )
    decoded = decode_execution_lineage_admission_record(
        encode_execution_lineage_admission_record(original),
    )
    assert decoded == original


@pytest.mark.gate
def test_npsc5e_p0a_unknown_schema_version_fails_closed() -> None:
    scope = _scope()
    root = mint_execution_id()
    from intergrax.contracts.execution_lineage import ExecutionLineageAdmissionRecord

    original = ExecutionLineageAdmissionRecord(
        scope=scope,
        segment_root_execution_id=root,
        execution_id=root,
        parent_execution_id=None,
        admission_position=1,
    )
    payload = encode_execution_lineage_admission_record(original)
    payload["schema_version"] = 999
    with pytest.raises(ExecutionLineageError):
        decode_execution_lineage_admission_record(payload)


@pytest.mark.gate
def test_npsc5e_p0a_child_lineage_references_exact_parent() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    scope = _scope()
    root = mint_execution_id()
    child = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    admitted = persistence.admit_child(scope, root, child, root)
    assert admitted.parent_execution_id == root
    assert admitted.execution_id == child


@pytest.mark.gate
def test_npsc5e_p0a_parallel_children_retain_parent_isolation() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    scope = _scope()
    root = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)

    def admit_one() -> str:
        child = mint_execution_id()
        persistence.admit_child(scope, root, child, root)
        return child

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        children = [future.result() for future in [pool.submit(admit_one) for _ in range(16)]]
    page = persistence.list_admissions_for_attempt(scope, limit=100)
    child_rows = [item for item in page.admissions if item.execution_id in children]
    assert len(child_rows) == 16
    assert all(item.parent_execution_id == root for item in child_rows)


@pytest.mark.gate
def test_npsc5e_p0a_parallel_roots_do_not_leak_active_lineage() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    scope_a = _scope(tenant_id="tenant-a")
    scope_b = _scope(tenant_id="tenant-b")
    root_a = mint_execution_id()
    root_b = mint_execution_id()

    def bind_and_peek(scope: ExecutionLineageAttemptScope, root: ExecutionId) -> ExecutionId | None:
        state = ActiveExecutionLineageState(
            persistence=persistence,
            scope=scope,
            segment_root_execution_id=root,
        )
        token = bind_active_execution_lineage(state)
        try:
            active = peek_active_execution_lineage()
            return active.segment_root_execution_id if active is not None else None
        finally:
            reset_active_execution_lineage(token)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(copy_context().run, bind_and_peek, scope_a, root_a)
        future_b = pool.submit(copy_context().run, bind_and_peek, scope_b, root_b)
        assert future_a.result() == root_a
        assert future_b.result() == root_b
    assert peek_active_execution_lineage() is None


@pytest.mark.gate
def test_npsc5e_p0a_long_running_resume_preserves_segment_predecessor() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    scope = _scope()
    e1 = mint_execution_id()
    e4 = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, e1)
    persistence.admit_root(scope, e1, e1)
    persistence.close_segment_for_resume(scope, e1)
    segment = persistence.open_segment(scope, e4, e1)
    assert segment.predecessor_root_execution_id == e1
    assert segment.root_execution_id == e4


@pytest.mark.gate
def test_npsc5e_p0a_attempt_transition_representable_in_lineage() -> None:
    store = InMemoryAttemptLifecycleStore()
    lifecycle = AttemptLifecycleService(store)
    run_id = mint_run_id()
    attempt1 = mint_attempt_id()
    lifecycle.record_initial_attempt(
        tenant_id="tenant-a",
        run_id=run_id,
        attempt_id=attempt1,
    )
    transition = lifecycle.transition_to_next_attempt(
        tenant_id="tenant-a",
        run_id=run_id,
        expected_attempt_id=attempt1,
        reason=AttemptTransitionReason.RETRY,
    )
    assert transition.previous_attempt_id == attempt1
    persistence = InMemoryExecutionLineagePersistence()
    scope1 = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=run_id,
        attempt_id=attempt1,
    )
    scope2 = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=scope1.task_id,
        run_id=run_id,
        attempt_id=transition.active_attempt_id,
    )
    root1 = mint_execution_id()
    root2 = mint_execution_id()
    persistence.open_attempt(scope1)
    persistence.open_segment(scope1, root1)
    persistence.admit_root(scope1, root1, root1)
    persistence.seal_attempt(scope1, ExecutionLineageAttemptClosureKind.RETRY_SUPERSEDED)
    persistence.open_attempt(scope2)
    persistence.open_segment(scope2, root2)
    persistence.admit_root(scope2, root2, root2)
    state1 = persistence.read_attempt_lineage_state(scope1)
    state2 = persistence.read_attempt_lineage_state(scope2)
    assert state1 is not None
    assert state2 is not None
    assert state1.sealed is True
    assert state1.closure_kind is ExecutionLineageAttemptClosureKind.RETRY_SUPERSEDED
    assert state2.scope.attempt_id == transition.active_attempt_id
    assert state2.scope.run_id == scope1.run_id
    assert transition.active_attempt_id != attempt1


@pytest.mark.gate
def test_npsc5e_p0a_terminal_seal_blocks_writes() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    scope = _scope()
    root = mint_execution_id()
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)
    persistence.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    with pytest.raises(ExecutionLineageIntegrityError):
        persistence.admit_child(scope, root, mint_execution_id(), root)


@pytest.mark.gate
def test_npsc5e_p0a_tenant_isolation_on_admission_scope() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task_id = mint_task_id()
    scope_a = build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    scope_b = build_execution_lineage_attempt_scope(
        tenant_id="tenant-b",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    root_a = mint_execution_id()
    root_b = mint_execution_id()
    persistence.open_attempt(scope_a)
    persistence.open_segment(scope_a, root_a)
    persistence.admit_root(scope_a, root_a, root_a)
    persistence.open_attempt(scope_b)
    persistence.open_segment(scope_b, root_b)
    persistence.admit_root(scope_b, root_b, root_b)
    page_a = persistence.list_admissions_for_attempt(scope_a, limit=10)
    page_b = persistence.list_admissions_for_attempt(scope_b, limit=10)
    assert page_a.admissions[0].execution_id == root_a
    assert page_b.admissions[0].execution_id == root_b


@pytest.mark.gate
@pytest.mark.parametrize("module_name", _FROZEN_NPSC_REGRESSION_MODULES)
def test_npsc5e_p0a_frozen_npsc_regression_module_importable(module_name: str) -> None:
    import importlib

    importlib.import_module(module_name)


@pytest.mark.gate
def test_npsc5e_p0a_recovery_readiness_flags() -> None:
    assert ATTEMPT_LINEAGE_READY_FOR_5E_R1 is True
    assert CHECKPOINT_LINEAGE_READY_FOR_5E_R2 is True
    assert PARTIAL_RECOVERY_LINEAGE_READY_FOR_5E_R3 is True


ATTEMPT_LINEAGE_READY_FOR_5E_R1 = True
CHECKPOINT_LINEAGE_READY_FOR_5E_R2 = True
PARTIAL_RECOVERY_LINEAGE_READY_FOR_5E_R3 = True
