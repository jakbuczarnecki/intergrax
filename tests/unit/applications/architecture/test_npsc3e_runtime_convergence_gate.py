# © Artur Czarnecki. All rights reserved.

"""NPSC-3E: debug/lab runtime convergence to canonical host execution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEBUG_ROOT = _REPO_ROOT / "intergrax" / "debug"
_LAB_ROOT = _REPO_ROOT / "applications" / "lab_application"
_ORG_WORKER_LAB = _REPO_ROOT / "intergrax" / "lab" / "organization_worker.py"
_FORBIDDEN_TOKENS = (
    "NexusLoopTaskExecutor",
    "UnifiedTaskRunner",
    "execute_root_task",
)


def _scan_paths(roots: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "tests" in path.parts:
                continue
            if "docker" in path.parts and "runtime-context" in path.parts:
                continue
            paths.append(path)
    return paths


def _token_violations(path: Path, tokens: tuple[str, ...]) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return []
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return [
        f"{rel}:{lineno}: forbidden token {token!r}"
        for token in tokens
        for lineno, line in enumerate(source.splitlines(), start=1)
        if token in line
    ]


def test_npsc3e_debug_runtime_has_no_legacy_executor_tokens() -> None:
    violations: list[str] = []
    for path in _scan_paths((_DEBUG_ROOT,)):
        violations.extend(_token_violations(path, _FORBIDDEN_TOKENS))
    assert violations == [], "Debug runtime still references legacy execution:\n" + "\n".join(violations)


def test_npsc3e_lab_application_has_no_legacy_executor_tokens() -> None:
    violations: list[str] = []
    for path in _scan_paths((_LAB_ROOT,)):
        violations.extend(_token_violations(path, _FORBIDDEN_TOKENS))
    assert violations == [], "Lab application still references legacy execution:\n" + "\n".join(violations)


def test_npsc3e_organization_worker_lab_uses_canonical_host_execution() -> None:
    source = _ORG_WORKER_LAB.read_text(encoding="utf-8")
    assert "HostTaskExecutionExecutor" in source
    assert "build_host_task_execution" in source
    assert "UnifiedTaskRunner" not in source
    assert "execute_root_task" not in source


def test_npsc3e_debug_app_wires_host_task_execution_executor() -> None:
    source = (_DEBUG_ROOT / "app.py").read_text(encoding="utf-8")
    assert "HostTaskExecutionExecutor" in source
    assert "build_host_task_execution" in source
    assert "UnifiedTaskRunner" not in source


def test_npsc3e_hitl_service_routes_through_host_execution() -> None:
    source = (_DEBUG_ROOT / "hitl_service.py").read_text(encoding="utf-8")
    assert "HostTaskExecutionPort" in source
    assert "execute_root_task" not in source
    assert "mint_run_id" not in source
    assert "mint_attempt_id" not in source
    assert "mint_execution_id" not in source
