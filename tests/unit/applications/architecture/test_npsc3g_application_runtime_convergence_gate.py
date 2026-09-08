# © Artur Czarnecki. All rights reserved.

"""NPSC-3G: application runtime execution convergence to canonical host task execution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"
_SHARED_QUEUE_WIRING = _REPO_ROOT / "intergrax" / "applications" / "_shared" / "queue_worker_wiring.py"
_RUN_ADAPTER = _REPO_ROOT / "intergrax" / "runtime" / "task" / "host_task_execution_run_adapter.py"
_QUEUED_ADAPTER = _REPO_ROOT / "intergrax" / "runtime" / "task" / "queued_host_task_execution_adapter.py"
_LEGACY_INLINE_ADAPTER = _REPO_ROOT / "intergrax" / "runtime" / "task" / "nexus_task_execution_adapter.py"
_LEGACY_QUEUED_ADAPTER = _REPO_ROOT / "intergrax" / "runtime" / "task" / "queued_nexus_execution_adapter.py"

_FORBIDDEN_FACTORY_TOKENS = (
    "NexusTaskExecutionAdapter",
    "QueuedNexusExecutionAdapter",
    "build_task_runner_with_enricher",
    "task_runner=task_runner",
)


def _factory_paths() -> list[Path]:
    paths: list[Path] = []
    for path in _APPLICATIONS_ROOT.rglob("host/factory.py"):
        if "__pycache__" in path.parts:
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


def test_npsc3g_tier3_factories_have_no_legacy_run_execution_adapters() -> None:
    violations: list[str] = []
    for path in _factory_paths():
        violations.extend(_token_violations(path, _FORBIDDEN_FACTORY_TOKENS))
    assert violations == [], "Tier-3 factories still reference legacy run execution:\n" + "\n".join(
        violations
    )


def test_npsc3g_queue_worker_wiring_routes_through_host_execution() -> None:
    source = _SHARED_QUEUE_WIRING.read_text(encoding="utf-8")
    assert "HostTaskExecutionRunAdapter" in source
    assert "QueuedHostTaskExecutionAdapter" in source
    assert "host_execution: HostTaskExecutionPort" in source
    assert "NexusTaskExecutionAdapter" not in source
    assert "QueuedNexusExecutionAdapter" not in source
    assert "task_runner:" not in source


def test_npsc3g_canonical_run_adapters_exist() -> None:
    assert _RUN_ADAPTER.is_file()
    assert _QUEUED_ADAPTER.is_file()
    run_source = _RUN_ADAPTER.read_text(encoding="utf-8")
    assert "HostTaskExecutionExecutor" in run_source
    assert "HostTaskExecutionPort" in run_source
    assert "UnifiedTaskRunner" not in run_source
    assert "execute_root_task" not in run_source


def test_npsc3g_legacy_nexus_run_adapters_retired() -> None:
    assert not _LEGACY_INLINE_ADAPTER.is_file()
    assert not _LEGACY_QUEUED_ADAPTER.is_file()


def test_npsc3g_worker_runtime_uses_host_execution() -> None:
    source = (
        _REPO_ROOT / "intergrax" / "runtime" / "task" / "nexus_worker_execution.py"
    ).read_text(encoding="utf-8")
    assert "HostTaskExecutionPort" in source
    assert "build_host_task_execution" in source
    assert "UnifiedTaskRunner" not in source
