# © Artur Czarnecki. All rights reserved.

"""EEC-1 — Execution Engine cross-plane certification architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    ExecutionId,
    RunId,
    TaskId,
    validate_execution_id,
    validate_run_id,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationTopologyExecutionId,
    mint_orchestration_topology_execution_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_EXECUTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution"
_EVIDENCE_ROOTS = (
    _REPO_ROOT / "intergrax" / "runtime" / "events",
    _REPO_ROOT / "intergrax" / "runtime" / "observability",
    _REPO_ROOT / "intergrax" / "contracts" / "execution_evidence",
)
_DIAGNOSTICS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
_RECOVERY_ROOTS = (
    _EXECUTION_ROOT / "retry",
    _EXECUTION_ROOT / "attempt_lifecycle",
    _EXECUTION_ROOT / "fan_out_partial_recovery.py",
)
_RECOVERY_CONTRACTS = (
    _REPO_ROOT / "intergrax" / "contracts" / "execution_retry.py",
    _REPO_ROOT / "intergrax" / "contracts" / "partial_recovery.py",
    _REPO_ROOT / "intergrax" / "contracts" / "recovery_admission.py",
)
_SCALE_CONTRACTS = (
    _REPO_ROOT / "intergrax" / "contracts" / "execution_capacity_admission.py",
    _REPO_ROOT / "intergrax" / "contracts" / "dependency_concurrency_admission.py",
    _REPO_ROOT / "intergrax" / "contracts" / "concurrent_execution_work.py",
    _REPO_ROOT / "intergrax" / "contracts" / "resilience_policy.py",
)
_SCALE_RUNTIME_SNIPPETS = (
    _EXECUTION_ROOT / "local_execution_capacity_admission.py",
    _EXECUTION_ROOT / "runtime.py",
    _REPO_ROOT / "intergrax" / "runtime" / "resilience",
)

_CANONICAL_ID_TYPES = frozenset(
    {"TaskId", "RunId", "AttemptId", "ExecutionId", "EventId"},
)

_FORBIDDEN_ALTERNATE_EXECUTION_ID_NAMES = frozenset(
    {
        "WorkerId",
        "NodeId",
        "WorkerExecutionId",
        "RunNodeId",
    },
)

_EXECUTION_CORE_FILES = frozenset(
    {
        "runtime.py",
        "boundary.py",
        "host_task.py",
        "child.py",
        "orchestration.py",
        "nexus_host_execution.py",
    },
)

_CANONICAL_EVIDENCE_WIRING_FILES = frozenset(
    {
        "host_task.py",
        "orchestration.py",
        "nexus_host_execution.py",
        "child.py",
        "runtime.py",
    },
)

_EVENT_STORE_MODULE_PREFIXES = (
    "intergrax.runtime.events.stores.",
)

_EXECUTION_RUNTIME_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.events.stores.",
)

_RECOVERY_FORBIDDEN_EVIDENCE_JOURNAL_SYMBOLS = frozenset(
    {
        "SQLiteRuntimeEventStore",
        "InMemoryRuntimeEventStore",
        "build_unified_run_journal",
        "ValidatingRuntimeEventPersistence",
    },
)

_EVIDENCE_FORBIDDEN_RECOVERY_IMPORT_PREFIXES = (
    "intergrax.runtime.execution.retry",
    "intergrax.runtime.execution.attempt_lifecycle",
    "intergrax.runtime.execution.fan_out_partial_recovery",
)

_EVIDENCE_FORBIDDEN_RECOVERY_SYMBOLS = frozenset(
    {
        "AttemptLifecycleService",
        "FanOutPartialRecoveryService",
        "evaluate_execution_retry_eligibility",
        "transition_to_next_attempt",
    },
)

_DIAGNOSTICS_FORBIDDEN_LIFECYCLE_SYMBOLS = frozenset(
    {
        "start_execution",
        "begin_execution",
        "ExecutionRuntime",
        "AttemptLifecycleService",
        "transition_to_next_attempt",
    },
)

_EVENTS_FORBIDDEN_EXECUTION_CONTROL_PREFIXES = (
    "intergrax.runtime.execution.runtime",
    "intergrax.runtime.execution.boundary",
    "intergrax.runtime.execution.retry",
    "intergrax.runtime.execution.attempt_lifecycle",
)

_RESILIENCE_FORBIDDEN_LIFECYCLE_SYMBOLS = frozenset(
    {
        "start_execution",
        "begin_execution",
        "ExecutionRuntime.execute",
        "mint_root_execution_identity",
        "activate_root_execution_lineage",
    },
)


def _iter_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    paths: list[Path] = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        paths.append(path)
    return paths


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _collect_import_module_prefixes(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


def _collect_newtype_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            if isinstance(node.value, ast.Call) and _call_name(node.value) == "NewType":
                names.add(node.targets[0].id)
    return names


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_call_symbols(tree: ast.AST) -> set[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name:
            symbols.add(name)
    return symbols


def _violations_for_import_prefixes(
    paths: list[Path],
    forbidden_prefixes: tuple[str, ...],
) -> list[str]:
    violations: list[str] = []
    for path in paths:
        tree = _parse(path)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in sorted(_collect_import_module_prefixes(tree)):
            for prefix in forbidden_prefixes:
                if module == prefix.rstrip(".") or module.startswith(prefix):
                    violations.append(f"{rel}: imports {module}")
    return violations


def _violations_for_symbols(paths: list[Path], forbidden: frozenset[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        tree = _parse(path)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for symbol in sorted(_collect_call_symbols(tree) & forbidden):
            violations.append(f"{rel}: calls {symbol}")
    return violations


@pytest.mark.gate
def test_eec1_execution_identity_uses_single_canonical_contract() -> None:
    assert TaskId.__name__ == "TaskId"
    assert RunId.__name__ == "RunId"
    assert AttemptId.__name__ == "AttemptId"
    assert ExecutionId.__name__ == "ExecutionId"
    assert EventId.__name__ == "EventId"


@pytest.mark.gate
def test_eec1_execution_runtime_does_not_define_parallel_identity_newtypes() -> None:
    violations: list[str] = []
    for path in _iter_python_files(_EXECUTION_ROOT):
        extra = _collect_newtype_names(_parse(path)) - _CANONICAL_ID_TYPES
        if extra:
            rel = path.relative_to(_REPO_ROOT).as_posix()
            violations.append(f"{rel}: {sorted(extra)}")
    assert violations == []


@pytest.mark.gate
def test_eec1_no_worker_or_node_execution_id_aliases_in_scoped_planes() -> None:
    scan_roots = (
        _EXECUTION_ROOT,
        *_EVIDENCE_ROOTS,
        _DIAGNOSTICS_ROOT,
    )
    violations: list[str] = []
    for root in scan_roots:
        for path in _iter_python_files(root):
            tree = _parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in _FORBIDDEN_ALTERNATE_EXECUTION_ID_NAMES:
                    rel = path.relative_to(_REPO_ROOT).as_posix()
                    violations.append(f"{rel}:{node.lineno}:{node.id}")
    assert violations == []


@pytest.mark.gate
def test_eec1_orchestration_topology_execution_id_not_run_or_worker_substitute() -> None:
    sample_run = validate_run_id("run_" + "a" * 32)
    sample_exec = validate_execution_id("exec_" + "b" * 32)
    assert sample_run != sample_exec
    topology_id = OrchestrationTopologyExecutionId(f"{sample_run}:attempt_{'c' * 32}:host:slot")
    assert not str(topology_id).startswith("exec_")
    assert str(topology_id).startswith("run_")
    assert "WorkerId" not in str(topology_id)


@pytest.mark.gate
def test_eec1_canonical_strategy_modules_wire_failure_evidence_contract() -> None:
    missing: list[str] = []
    for name in _CANONICAL_EVIDENCE_WIRING_FILES:
        path = _EXECUTION_ROOT / name
        source = path.read_text(encoding="utf-8")
        if (
            "ExecutionFailureEvidenceRecorder" not in source
            and "wrap_execution_delegate_for_failure_evidence" not in source
            and "failure_evidence_recorder" not in source
        ):
            missing.append(name)
    assert missing == []


_EVIDENCE_CONTRACT_FORBIDDEN_RUNTIME_PREFIXES = (
    "intergrax.runtime.execution.",
    "intergrax.runtime.resilience.",
    "intergrax.runtime.events.stores.",
)


@pytest.mark.gate
def test_eec1_execution_evidence_contracts_do_not_import_runtime_planes() -> None:
    violations: list[str] = []
    for path in _iter_python_files(_REPO_ROOT / "intergrax" / "contracts" / "execution_evidence"):
        tree = _parse(path)
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_import_module_prefixes(tree):
            for prefix in _EVIDENCE_CONTRACT_FORBIDDEN_RUNTIME_PREFIXES:
                if module.startswith(prefix):
                    violations.append(f"{rel}: {module}")
    assert violations == []


@pytest.mark.gate
def test_eec1_recovery_plane_does_not_own_parallel_event_journal() -> None:
    paths: list[Path] = []
    for root in _RECOVERY_ROOTS:
        paths.extend(_iter_python_files(root))
    for contract in _RECOVERY_CONTRACTS:
        paths.append(contract)
    violations: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for symbol in _RECOVERY_FORBIDDEN_EVIDENCE_JOURNAL_SYMBOLS:
            if symbol in source:
                violations.append(f"{rel}: references {symbol}")
        for prefix in _EVENT_STORE_MODULE_PREFIXES:
            if prefix in source:
                violations.append(f"{rel}: imports store prefix {prefix}")
    assert violations == []


@pytest.mark.gate
def test_eec1_evidence_plane_does_not_import_recovery_implementation() -> None:
    paths: list[Path] = []
    for root in _EVIDENCE_ROOTS:
        paths.extend(_iter_python_files(root))
    violations = _violations_for_import_prefixes(
        paths,
        _EVIDENCE_FORBIDDEN_RECOVERY_IMPORT_PREFIXES,
    )
    violations.extend(_violations_for_symbols(paths, _EVIDENCE_FORBIDDEN_RECOVERY_SYMBOLS))
    assert violations == []


@pytest.mark.gate
def test_eec1_diagnostics_reconstruction_does_not_mutate_execution_lifecycle() -> None:
    paths = _iter_python_files(_DIAGNOSTICS_ROOT)
    violations = _violations_for_symbols(paths, _DIAGNOSTICS_FORBIDDEN_LIFECYCLE_SYMBOLS)
    assert violations == []


@pytest.mark.gate
def test_eec1_events_reconstruction_does_not_import_execution_control() -> None:
    paths = _iter_python_files(_REPO_ROOT / "intergrax" / "runtime" / "events")
    violations = _violations_for_import_prefixes(paths, _EVENTS_FORBIDDEN_EXECUTION_CONTROL_PREFIXES)
    assert violations == []


@pytest.mark.gate
def test_eec1_execution_runtime_core_does_not_import_event_store_implementations() -> None:
    violations: list[str] = []
    for name in _EXECUTION_CORE_FILES:
        path = _EXECUTION_ROOT / name
        violations.extend(
            _violations_for_import_prefixes([path], _EXECUTION_RUNTIME_FORBIDDEN_IMPORT_PREFIXES),
        )
    assert violations == []


@pytest.mark.gate
def test_eec1_capacity_admission_precedes_boundary_work_in_runtime() -> None:
    source = (_EXECUTION_ROOT / "runtime.py").read_text(encoding="utf-8")
    acquire_idx = source.index("await self._execution_capacity_admission.acquire")
    boundary_idx = source.index("boundary = ExecutionBoundary")
    assert acquire_idx < boundary_idx


@pytest.mark.gate
def test_eec1_scale_contracts_remain_admission_and_policy_only() -> None:
    violations: list[str] = []
    for path in _SCALE_CONTRACTS:
        tree = _parse(path)
        for module in _collect_import_module_prefixes(tree):
            if module.startswith("intergrax.runtime.execution."):
                rel = path.relative_to(_REPO_ROOT).as_posix()
                violations.append(f"{rel}: {module}")
    assert violations == []


@pytest.mark.gate
def test_eec1_resilience_handoff_does_not_drive_execution_lifecycle() -> None:
    paths = _iter_python_files(_REPO_ROOT / "intergrax" / "runtime" / "resilience")
    violations: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for symbol in _RESILIENCE_FORBIDDEN_LIFECYCLE_SYMBOLS:
            if symbol in source:
                violations.append(f"{rel}: references {symbol}")
    assert violations == []


@pytest.mark.gate
def test_eec1_mint_orchestration_topology_execution_id_requires_active_run_attempt() -> None:
    from intergrax.contracts.execution_identity import bind_active_execution_identity

    run_id = validate_run_id("run_" + "d" * 32)
    attempt_id = AttemptId("attempt_" + "e" * 32)
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        from intergrax.contracts.orchestration_topology import (
            OrchestrationSlot,
            OrchestrationTopology,
        )

        topology: OrchestrationTopology[object] = OrchestrationTopology(
            slots=(OrchestrationSlot(slot_id="s1", payload=object()),),
        )
        derived = mint_orchestration_topology_execution_id(
            host_task_id="host-task",
            topology=topology,
        )
        assert str(run_id) in str(derived)
        assert str(attempt_id) in str(derived)
    finally:
        from intergrax.contracts.execution_identity import reset_active_execution_identity

        reset_active_execution_identity(token)
