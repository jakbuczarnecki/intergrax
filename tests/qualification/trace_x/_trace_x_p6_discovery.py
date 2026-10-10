# © Artur Czarnecki. All rights reserved.

"""Closed-world discovery for TRACE-X-P6 restart/resume and terminal producers."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DISCOVERY_ROOTS = (
    _REPO_ROOT / "intergrax",
    _REPO_ROOT / "applications",
    _REPO_ROOT / "agents",
)

_EXCLUDE_PATH_PARTS = (
    "/tests/",
    "\\tests\\",
    "/test_",
    "\\test_",
    "/qualification/",
    "\\qualification\\",
    "/examples/",
    "\\examples\\",
    "/scaffold/",
    "\\scaffold\\",
    "docker/runtime-context",
)

_RESTART_RESUME_MARKERS: tuple[str, ...] = (
    "TaskCheckpoint",
    "ExecutionContinuation",
    "UnifiedTaskResumeExecutor",
    "SQLiteTaskCheckpointStore",
    "LongRunningScheduler",
    "resume_token",
    "PendingExecutionContinuation",
    "restart_qualification",
    "assert_checkpoint_resumable",
    "ExecutionTerminalService",
    "commit_terminal_outcome",
    "worker_recovery",
    "WorkerQualifiedCapabilityResume",
    "redeliver",
    "RetryCoordinator",
    "fan_out_partial_recovery",
    "CheckpointStoreExecutionTerminalStore",
    "should_resume_acp_checkpoint",
)

_TERMINAL_MARKERS: tuple[str, ...] = (
    "commit_terminal_outcome",
    "ExecutionTerminalOutcome",
    "ExecutionTerminalService",
    "reconcile_task_state_with_terminal",
    "terminal_outcome",
    "ExecutionTerminalRecord",
    "record_cancellation",
    "ExecutionTerminalConflictError",
)


def _normalize_repo_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_discovery_candidate(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    normalized = path.as_posix()
    for part in _EXCLUDE_PATH_PARTS:
        if part.replace("/", "\\") in normalized or part in normalized:
            return False
    return True


def discover_restart_resume_path_keys() -> frozenset[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_discovery_candidate(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not any(marker in text for marker in _RESTART_RESUME_MARKERS):
                continue
            keys.add((_normalize_repo_path(path), "module"))
    return frozenset(keys)


def discover_terminal_producer_keys() -> frozenset[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_discovery_candidate(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if not any(marker in text for marker in _TERMINAL_MARKERS):
                continue
            keys.add((_normalize_repo_path(path), "module"))
    return frozenset(keys)


def grep_production_pattern(pattern: str) -> list[str]:
    import re

    hits: list[str] = []
    compiled = re.compile(pattern)
    for root in _DISCOVERY_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if not _is_discovery_candidate(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if compiled.search(text):
                hits.append(_normalize_repo_path(path))
    return sorted(hits)
