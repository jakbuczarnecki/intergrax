# © Artur Czarnecki. All rights reserved.

"""Mechanical semantic-owner discovery for TRACE-X-P6 (independent of matrix prose)."""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=128)
def _module_defines_class(repo_relative_path: str, class_name: str) -> bool:
    path = _REPO_ROOT / repo_relative_path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body
    )


def _anchor_owner(class_name: str, repo_relative_path: str) -> frozenset[str]:
    if not _module_defines_class(repo_relative_path, class_name):
        return frozenset()
    return frozenset({repo_relative_path})


def discover_semantic_owners(concern: str) -> frozenset[str]:
    """Return canonical owner anchor module(s) for a named P6 concern."""
    key = concern.strip().lower()
    if key == "execution identity owner":
        return frozenset({"intergrax/contracts/execution_identity.py"})
    if key == "run identity owner":
        return frozenset({"intergrax/runtime/execution/identity_authority.py"})
    if key == "attempt identity owner":
        return _anchor_owner(
            "AttemptLifecycleService",
            "intergrax/runtime/execution/attempt_lifecycle/service.py",
        )
    if key == "checkpoint persistence owner":
        return _anchor_owner(
            "SQLiteTaskCheckpointStore",
            "intergrax/runtime/long_running/store.py",
        )
    if key == "resumability decision owner":
        path = "intergrax/runtime/cancellation/resume_admission.py"
        if _module_defines_class(path, "CheckpointNotResumableError"):
            return frozenset({path})
        return frozenset()
    if key == "resume coordination owner":
        return _anchor_owner(
            "LongRunningScheduler",
            "intergrax/runtime/long_running/scheduler.py",
        )
    if key == "resume admission tenant/identity validation owner":
        resume = "intergrax/runtime/cancellation/resume_admission.py"
        reentry = "intergrax/runtime/background_execution/reentry_admission.py"
        owners: set[str] = set()
        if _module_defines_class(resume, "CheckpointNotResumableError"):
            owners.add(resume)
        if _module_defines_class(reentry, "BackgroundExecutionReentryAdmissionError"):
            owners.add(reentry)
        return frozenset(owners)
    if key == "retry policy owner":
        return frozenset({"intergrax/runtime/execution/retry/policy.py"})
    if key == "retry orchestration owner":
        return _anchor_owner(
            "RetryCoordinator",
            "intergrax/runtime/nexus/retry/coordinator.py",
        )
    if key == "terminal state truth owner":
        return _anchor_owner(
            "ExecutionTerminalService",
            "intergrax/runtime/execution/execution_terminal/service.py",
        )
    if key == "terminal runtimeevent/evidence owner":
        return frozenset({"intergrax/runtime/events/trace_bridge.py"})
    if key == "failure reconstruction owner":
        return _anchor_owner(
            "ExecutionReconstructor",
            "intergrax/runtime/observability/reconstruction/execution_reconstruction.py",
        )
    if key == "parent-child causality owner":
        return _anchor_owner(
            "ExecutionLineagePersistence",
            "intergrax/contracts/execution_lineage.py",
        )
    raise KeyError(f"unknown P6 semantic owner concern: {concern}")


P6_SEMANTIC_OWNER_MATRIX: Final[tuple[tuple[str, int], ...]] = (
    ("Execution identity owner", 1),
    ("Run identity owner", 1),
    ("Attempt identity owner", 1),
    ("checkpoint persistence owner", 1),
    ("resumability decision owner", 1),
    ("resume coordination owner", 1),
    ("resume admission tenant/identity validation owner", 2),
    ("retry policy owner", 1),
    ("retry orchestration owner", 1),
    ("terminal state truth owner", 1),
    ("terminal RuntimeEvent/evidence owner", 1),
    ("failure reconstruction owner", 1),
    ("parent-child causality owner", 1),
)
