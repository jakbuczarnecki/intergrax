# © Artur Czarnecki. All rights reserved.

"""NPSC-4.1 — execution boundary hardening architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"
_SHARED_APPLICATIONS_ROOT = _REPO_ROOT / "intergrax" / "applications"
_INTERACTIONS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "interactions"
_TASK_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "task"
_NEXUS_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "nexus"
_SCHEDULER_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "long_running"
_EXECUTION_RUNTIME_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "runtime.py"
_IDENTITY_AUTHORITY_PATH = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "identity_authority.py"
_TASK_RUN_BRIDGE_PATH = _TASK_ROOT / "task_run_bridge.py"
_UNIFIED_TASK_RUNNER_PATH = _TASK_ROOT / "unified_task_runner.py"

_IDENTITY_INTAKE_FORBIDDEN_CALLS = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
        "mint_intake_execution_identity",
        "new_run_id",
    }
)

_IDENTITY_BIND_CALL = "bind_active_execution_identity"

_LIFECYCLE_CALLS = frozenset({"start_execution", "begin_execution"})

_INTAKE_EXCLUDED_PATH_PARTS = frozenset({"docker", "runtime-context", "tests", "__pycache__"})

_INTAKE_EXCLUDED_SHARED_FILES = frozenset(
    {
        "intergrax/applications/_shared/harness_task_routes.py",
        "intergrax/applications/_shared/scenario_runtime_baseline.py",
    }
)

_LIFECYCLE_ALLOWED_FILES = frozenset(
    {
        (_REPO_ROOT / "intergrax" / "runtime" / "task" / "host_task_execution_run_adapter.py")
        .relative_to(_REPO_ROOT)
        .as_posix(),
        (_REPO_ROOT / "intergrax" / "runtime" / "task" / "queued_host_task_execution_adapter.py")
        .relative_to(_REPO_ROOT)
        .as_posix(),
        (_REPO_ROOT / "intergrax" / "fastapi_core" / "execution" / "adapters" / "adapter.py")
        .relative_to(_REPO_ROOT)
        .as_posix(),
        (_REPO_ROOT / "intergrax" / "fastapi_core" / "execution" / "adapters" / "default_adapter.py")
        .relative_to(_REPO_ROOT)
        .as_posix(),
        (_REPO_ROOT / "intergrax" / "fastapi_core" / "execution" / "adapters" / "inprocess_adapter.py")
        .relative_to(_REPO_ROOT)
        .as_posix(),
        (_REPO_ROOT / "intergrax" / "fastapi_core" / "execution" / "adapters" / "threaded_adapter.py")
        .relative_to(_REPO_ROOT)
        .as_posix(),
    }
)

_UNIFIED_TASK_RUNNER_HARNESS_ALLOWLIST_ROOTS = (
    _REPO_ROOT / "intergrax" / "applications" / "_shared",
    _REPO_ROOT / "intergrax" / "runtime" / "long_running",
    _REPO_ROOT / "intergrax" / "eval",
    _REPO_ROOT / "intergrax" / "experiments",
    _REPO_ROOT / "intergrax" / "harness",
    _REPO_ROOT / "intergrax" / "scaffold",
)

_FORBIDDEN_INTAKE_SCAN_ROOTS = (
    _APPLICATIONS_ROOT,
    _SHARED_APPLICATIONS_ROOT,
    _INTERACTIONS_ROOT,
    _NEXUS_ROOT,
    _SCHEDULER_ROOT,
)


def _path_excluded(path: Path) -> bool:
    return any(part in _INTAKE_EXCLUDED_PATH_PARTS for part in path.parts)


def _iter_intake_python_files() -> list[Path]:
    paths: list[Path] = []
    for path in _APPLICATIONS_ROOT.rglob("*.py"):
        if _path_excluded(path):
            continue
        if "serving" not in path.parts and path.name != "workspace_routes.py":
            continue
        paths.append(path)
    shared_mcp = _SHARED_APPLICATIONS_ROOT / "_shared" / "mcp_nexus_server.py"
    if shared_mcp.is_file():
        paths.append(shared_mcp)
    return paths


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _iter_python_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*.py"):
        if _path_excluded(path):
            continue
        paths.append(path)
    return paths


def _collect_intake_forbidden_calls(
    *,
    forbidden: frozenset[str],
) -> list[str]:
    violations: list[str] = []
    for path in _iter_intake_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _INTAKE_EXCLUDED_SHARED_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in forbidden:
                violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def _collect_forbidden_calls(
    roots: tuple[Path, ...],
    *,
    forbidden: frozenset[str],
    allowed_files: frozenset[str] = frozenset(),
) -> list[str]:
    violations: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in _iter_python_files(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in allowed_files:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                if name in forbidden:
                    violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def test_npsc4_1_task_run_bridge_has_no_intake_identity_helper() -> None:
    source = _TASK_RUN_BRIDGE_PATH.read_text(encoding="utf-8")
    assert "mint_intake_execution_identity" not in source


def test_npsc4_1_application_intake_does_not_mint_execution_identity() -> None:
    violations = _collect_intake_forbidden_calls(forbidden=_IDENTITY_INTAKE_FORBIDDEN_CALLS)
    assert violations == [], (
        "application intake must not mint execution identity: " + ", ".join(violations)
    )


def test_npsc4_1_interactions_do_not_mint_execution_identity() -> None:
    violations = _collect_forbidden_calls(
        (_INTERACTIONS_ROOT,),
        forbidden=_IDENTITY_INTAKE_FORBIDDEN_CALLS,
    )
    assert violations == [], (
        "interaction intake must not mint execution identity: " + ", ".join(violations)
    )


def test_npsc4_1_nexus_does_not_mint_or_bind_execution_identity() -> None:
    mint_violations = _collect_forbidden_calls(
        (_NEXUS_ROOT,),
        forbidden=_IDENTITY_INTAKE_FORBIDDEN_CALLS | frozenset({"mint_task_id"}),
    )
    bind_violations = _collect_forbidden_calls(
        (_NEXUS_ROOT,),
        forbidden=frozenset({_IDENTITY_BIND_CALL}),
    )
    assert mint_violations == [], "Nexus must not mint execution identity: " + ", ".join(mint_violations)
    assert bind_violations == [], "Nexus must not bind execution identity: " + ", ".join(bind_violations)


def test_npsc4_1_scheduler_does_not_mint_execution_identity() -> None:
    violations = _collect_forbidden_calls(
        (_SCHEDULER_ROOT,),
        forbidden=_IDENTITY_INTAKE_FORBIDDEN_CALLS | frozenset({"mint_task_id"}),
    )
    assert violations == [], (
        "scheduler must not mint execution identity: " + ", ".join(violations)
    )


def test_npsc4_1_forbidden_zones_do_not_bind_execution_identity() -> None:
    violations = _collect_forbidden_calls(
        _FORBIDDEN_INTAKE_SCAN_ROOTS,
        forbidden=frozenset({_IDENTITY_BIND_CALL}),
    )
    assert violations == [], (
        "bind_active_execution_identity must remain owned by ExecutionBoundary: "
        + ", ".join(violations)
    )


def test_npsc4_1_forbidden_zones_do_not_start_execution_lifecycle() -> None:
    violations = _collect_forbidden_calls(
        _FORBIDDEN_INTAKE_SCAN_ROOTS,
        forbidden=_LIFECYCLE_CALLS,
    )
    scoped = [item for item in violations if item.split(":")[0] not in _LIFECYCLE_ALLOWED_FILES]
    assert scoped == [], (
        "lifecycle start must remain owned by ExecutionRuntime adapters: " + ", ".join(scoped)
    )


def test_npsc4_1_unified_task_runner_classified_harness_scheduling_only() -> None:
    source = _UNIFIED_TASK_RUNNER_PATH.read_text(encoding="utf-8")
    assert "HARNESS / SCHEDULING ONLY" in source
    assert "from intergrax.runtime.execution.host_task import HostTaskExecutionPort" not in source
    for token in ("mint_run_id", "mint_attempt_id", "mint_execution_id", "mint_task_id"):
        assert token not in source


def test_npsc4_1_tier3_factories_do_not_reference_unified_task_runner() -> None:
    violations: list[str] = []
    for path in _APPLICATIONS_ROOT.rglob("host/factory.py"):
        if _path_excluded(path):
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        if "UnifiedTaskRunner" in source:
            violations.append(f"{rel}: UnifiedTaskRunner")
    assert violations == [], (
        "Tier-3 factories must route through HostTaskExecutionPort, not UnifiedTaskRunner: "
        + ", ".join(violations)
    )


def test_npsc4_1_execution_runtime_remains_lifecycle_owner() -> None:
    source = _EXECUTION_RUNTIME_PATH.read_text(encoding="utf-8")
    assert "class ExecutionRuntime" in source
    assert "mint_root_execution_identity" in source


def test_npsc4_1_identity_authority_remains_sole_mint_module() -> None:
    source = _IDENTITY_AUTHORITY_PATH.read_text(encoding="utf-8")
    assert "def mint_root_execution_identity" in source
    assert "def mint_background_transport_identity" in source
