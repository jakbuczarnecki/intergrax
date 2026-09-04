# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-flight task registry for mid-run cancel and autonomy changes (FLOW-CTL)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

from intergrax.contracts.active_execution_task_scope import (
    ActiveExecutionTaskScopeUnavailable,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.task.task import Task

_ACTIVE: Dict[TaskId, ActiveTaskBinding] = {}
_LOCK = asyncio.Lock()


@dataclass(frozen=True, slots=True)
class ActiveTaskBinding:
    """Exact active execution ownership: one task_id + run_id pair."""

    task_id: TaskId
    run_id: RunId
    task: Task


class ActiveTaskOwnershipConflict(Exception):
    """Active binding already exists for task_id under a different run_id."""

    def __init__(
        self,
        task_id: TaskId,
        existing_run_id: RunId,
        requested_run_id: RunId,
    ) -> None:
        self.task_id = task_id
        self.existing_run_id = existing_run_id
        self.requested_run_id = requested_run_id
        super().__init__(
            f"active task ownership conflict for {task_id!r}: "
            f"existing run {existing_run_id!r}, requested {requested_run_id!r}"
        )


class ActiveTaskRegistry:
    @staticmethod
    async def register(task: Task, run_id: RunId) -> None:
        validated_run_id = validate_run_id(run_id)
        task_id = task.task_id
        async with _LOCK:
            existing = _ACTIVE.get(task_id)
            if existing is not None and existing.run_id != validated_run_id:
                raise ActiveTaskOwnershipConflict(
                    task_id,
                    existing.run_id,
                    validated_run_id,
                )
            _ACTIVE[task_id] = ActiveTaskBinding(
                task_id=task_id,
                run_id=validated_run_id,
                task=task,
            )

    @staticmethod
    async def unregister(task_id: TaskId | str, run_id: RunId) -> bool:
        validated_task_id = validate_task_id(task_id)
        validated_run_id = validate_run_id(run_id)
        async with _LOCK:
            existing = _ACTIVE.get(validated_task_id)
            if existing is None:
                return False
            if existing.run_id != validated_run_id:
                return False
            del _ACTIVE[validated_task_id]
            return True

    @staticmethod
    async def get(task_id: TaskId | str) -> Optional[ActiveTaskBinding]:
        validated_task_id = validate_task_id(task_id)
        async with _LOCK:
            return _ACTIVE.get(validated_task_id)

    @staticmethod
    async def list_ids() -> list[str]:
        async with _LOCK:
            return [str(task_id) for task_id in _ACTIVE]

    @staticmethod
    def peek_task_id_for_run(run_id: RunId) -> TaskId | None:
        """Process-local lookup of the active task owning ``run_id``."""
        validated_run_id = validate_run_id(run_id)
        for binding in _ACTIVE.values():
            if binding.run_id == validated_run_id:
                return binding.task_id
        return None

    @staticmethod
    def clear_for_tests() -> None:
        _ACTIVE.clear()


class ActiveTaskRegistryTaskScopeResolver:
    """Reference Production V1 resolver backed by in-flight task registration."""

    def resolve_current_task_scope(
        self,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> TaskId:
        del attempt_id, execution_id
        task_id = ActiveTaskRegistry.peek_task_id_for_run(run_id)
        if task_id is None:
            raise ActiveExecutionTaskScopeUnavailable(
                f"no active task scope for run {run_id!r}",
            )
        return task_id


__all__ = [
    "ActiveTaskBinding",
    "ActiveTaskOwnershipConflict",
    "ActiveTaskRegistry",
    "ActiveTaskRegistryTaskScopeResolver",
]
