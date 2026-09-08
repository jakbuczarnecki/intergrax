# © Artur Czarnecki. All rights reserved.

"""NPSC-3C-C: legacy NexusLoopTaskExecutor must not appear in active code paths."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FORBIDDEN_TOKEN = "NexusLoopTaskExecutor"

_SCAN_ROOTS: tuple[Path, ...] = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "applications",
)


def _scan_paths() -> list[Path]:
    paths: list[Path] = []
    for root in _SCAN_ROOTS:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "docker" in path.parts and "runtime-context" in path.parts:
                continue
            paths.append(path)
    return paths


def _violations_for_path(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return []
    if _FORBIDDEN_TOKEN not in source:
        return []
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return [
        f"{rel}:{lineno}: forbidden token {_FORBIDDEN_TOKEN!r}"
        for lineno, line in enumerate(source.splitlines(), start=1)
        if _FORBIDDEN_TOKEN in line
    ]


def test_npsc3c_legacy_nexus_loop_task_executor_is_fully_retired() -> None:
    violations: list[str] = []
    for path in _scan_paths():
        violations.extend(_violations_for_path(path))
    assert violations == [], "Legacy executor still referenced:\n" + "\n".join(violations)


def test_interaction_task_executor_exposes_canonical_host_adapter() -> None:
    source = (
        _REPO_ROOT / "intergrax" / "runtime" / "interactions" / "task_executor.py"
    ).read_text(encoding="utf-8")
    assert "class HostTaskExecutionExecutor" in source
    assert "HostTaskExecutionPort" in source
    assert "UnifiedTaskRunner" not in source
    assert "NexusLoop" not in source
