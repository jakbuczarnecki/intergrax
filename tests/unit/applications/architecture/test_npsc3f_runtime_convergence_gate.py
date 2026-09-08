# © Artur Czarnecki. All rights reserved.

"""NPSC-3F: harness runtime convergence to canonical host execution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HARNESS_ROOT = _REPO_ROOT / "intergrax" / "harness"
_SHARED_ROOT = _REPO_ROOT / "intergrax" / "applications" / "_shared"
_SCAFFOLD_ROOT = _REPO_ROOT / "intergrax" / "scaffold"
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"

_FORBIDDEN_TOKENS = (
    "build_harness_host_task_runner",
    "mount_harness_task_routes(",
    "wire_long_running_scheduler(",
)

_ALLOWLIST = frozenset(
    {
        _SHARED_ROOT / "harness_task_routes.py",
        _REPO_ROOT / "intergrax" / "runtime" / "long_running" / "wiring.py",
    }
)


def _scan_paths(roots: tuple[Path, ...]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
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
    if path in _ALLOWLIST:
        return []
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


def _collect_violations() -> list[str]:
    roots = (
        _HARNESS_ROOT,
        _SHARED_ROOT,
        _SCAFFOLD_ROOT,
    )
    violations: list[str] = []
    for path in _scan_paths(roots):
        violations.extend(_token_violations(path, _FORBIDDEN_TOKENS))
    for path in _scan_paths((_APPLICATIONS_ROOT,)):
        if path.name != "factory.py" or "host" not in path.parts:
            continue
        violations.extend(_token_violations(path, _FORBIDDEN_TOKENS))
    return violations


def test_npsc3f_harness_paths_have_no_legacy_execution_composition() -> None:
    violations = _collect_violations()
    assert violations == [], "Harness runtime still references legacy execution:\n" + "\n".join(violations)


def test_npsc3f_task_control_wiring_uses_canonical_host_execution() -> None:
    source = (_SHARED_ROOT / "task_control_wiring.py").read_text(encoding="utf-8")
    assert "mount_canonical_harness_task_routes" in source
    assert "HostTaskExecutionExecutor" in source
    assert "host_execution: HostTaskExecutionPort" in source
    assert "mount_harness_task_routes(" not in source


def test_npsc3f_lab_fastapi_routes_through_host_execution() -> None:
    source = (_HARNESS_ROOT / "lab_fastapi.py").read_text(encoding="utf-8")
    assert "HostTaskExecutionExecutor" in source
    assert "UnifiedTaskRunner" not in source


def test_npsc3f_harness_host_auxiliary_wiring_has_no_task_runner_factory() -> None:
    source = (_SHARED_ROOT / "harness_host_auxiliary_wiring.py").read_text(encoding="utf-8")
    assert "build_harness_host_task_runner" not in source
    assert "wire_harness_host_long_running_scheduler" in source
    assert "wire_harness_task_control" in source
