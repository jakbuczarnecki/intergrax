# © Artur Czarnecki. All rights reserved.

"""STATE-X-R4 — task checkpoint restore qualification support."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

STATE_X_R4_PRE_AUDIT_HEAD: Final[str] = "716ed746f8b463681db536804235cedc86adc162"

_REPO_ROOT = Path(__file__).resolve().parents[3]

RESTORE_CONSUMER_INVENTORY: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "LongRunningCoordinator.restore_if_resuming",
        "intergrax/runtime/long_running/coordinator.py",
        "assert_checkpoint_resume_eligible before Task.model_validate",
    ),
    (
        "build_checkpoint_resume_task",
        "intergrax/runtime/long_running/resume_planner.py",
        "assert_checkpoint_resume_materialization_eligible before materialization",
    ),
    (
        "build_scheduled_resume_task",
        "intergrax/runtime/long_running/resume_planner.py",
        "materialization gate via _base_resume_task",
    ),
    (
        "build_timeout_resume_task",
        "intergrax/runtime/long_running/resume_planner.py",
        "materialization gate via _base_resume_task",
    ),
    (
        "LongRunningScheduler",
        "intergrax/runtime/long_running/scheduler.py",
        "_can_materialize_checkpoint before Task build",
    ),
    (
        "governed_resume_checkpoint_task",
        "intergrax/applications/_shared/task_control.py",
        "materialization assert before HITL materialization",
    ),
    (
        "NexusWorkerRuntime._reconcile_resume_identity",
        "intergrax/runtime/task/nexus_worker_execution.py",
        "restore_if_resuming canonical validation",
    ),
    (
        "HostTaskResumeExecutor",
        "intergrax/runtime/long_running/scheduler.py",
        "consumes pre-validated Task from scheduler",
    ),
)


def resume_planner_validates_before_model_validate() -> bool:
    path = _REPO_ROOT / "intergrax/runtime/long_running/resume_planner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "_base_resume_task":
            continue
        segment = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
        return (
            "assert_checkpoint_resume_materialization_eligible" in segment
            and segment.find("assert_checkpoint_resume_materialization_eligible")
            < segment.find("Task.model_validate")
        )
    return False


def scheduler_validates_before_build() -> bool:
    path = _REPO_ROOT / "intergrax/runtime/long_running/scheduler.py"
    text = path.read_text(encoding="utf-8")
    return "_can_materialize_checkpoint" in text and "assert_checkpoint_resume_materialization_eligible" in text
