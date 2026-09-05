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

_TASK_BINDINGS: Dict[TaskId, ActiveTaskBinding] = {}
_TASK_BY_RUN: Dict[RunId, TaskId] = {}
_LOCK = asyncio.Lock()


@dataclass(frozen=True, slots=True)
class ActiveTaskBinding:
    """Exact active execution ownership: one task_id + run_id pair.

    The registry enforces bidirectional uniqueness: each active TaskId owns
    exactly one RunId and each active RunId belongs to exactly one TaskId.
    """

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


class ActiveRunOwnershipConflict(Exception):
    """Active binding already exists for run_id under a different task_id."""

    def __init__(
        self,
        run_id: RunId,
        existing_task_id: TaskId,
        requested_task_id: TaskId,
    ) -> None:
        self.run_id = run_id
        self.existing_task_id = existing_task_id
        self.requested_task_id = requested_task_id
        super().__init__(
            f"active run ownership conflict for {run_id!r}: "
            f"existing task {existing_task_id!r}, requested {requested_task_id!r}"
        )


class ActiveTaskRegistryInvariantError(RuntimeError):
    """Task and run authority indexes disagree."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _assert_task_binding_index_consistent(binding: ActiveTaskBinding) -> None:
    mapped_task_id = _TASK_BY_RUN.get(binding.run_id)
    if mapped_task_id != binding.task_id:
        raise ActiveTaskRegistryInvariantError(
            f"task binding for {binding.task_id!r} disagrees with run index "
            f"for {binding.run_id!r}: mapped task {mapped_task_id!r}"
        )


def _assert_run_index_consistent(run_id: RunId, task_id: TaskId) -> None:
    binding = _TASK_BINDINGS.get(task_id)
    if binding is None:
        raise ActiveTaskRegistryInvariantError(
            f"run index maps {run_id!r} to {task_id!r} but no task binding exists"
        )
    if binding.run_id != run_id:
        raise ActiveTaskRegistryInvariantError(
            f"run index maps {run_id!r} to {task_id!r} but binding run is "
            f"{binding.run_id!r}"
        )


class ActiveTaskRegistry:
    @staticmethod
    async def register(task: Task, run_id: RunId) -> None:
        validated_run_id = validate_run_id(run_id)
        task_id = task.task_id
        async with _LOCK:
            existing_by_task = _TASK_BINDINGS.get(task_id)
            existing_task_for_run = _TASK_BY_RUN.get(validated_run_id)

            if existing_by_task is not None:
                _assert_task_binding_index_consistent(existing_by_task)
            if existing_task_for_run is not None:
                _assert_run_index_consistent(validated_run_id, existing_task_for_run)

            if (
                existing_by_task is not None
                and existing_by_task.run_id != validated_run_id
            ):
                raise ActiveTaskOwnershipConflict(
                    task_id,
                    existing_by_task.run_id,
                    validated_run_id,
                )
            if existing_task_for_run is not None and existing_task_for_run != task_id:
                raise ActiveRunOwnershipConflict(
                    validated_run_id,
                    existing_task_for_run,
                    task_id,
                )

            _TASK_BINDINGS[task_id] = ActiveTaskBinding(
                task_id=task_id,
                run_id=validated_run_id,
                task=task,
            )
            _TASK_BY_RUN[validated_run_id] = task_id

    @staticmethod
    async def unregister(task_id: TaskId | str, run_id: RunId) -> bool:
        validated_task_id = validate_task_id(task_id)
        validated_run_id = validate_run_id(run_id)
        async with _LOCK:
            existing = _TASK_BINDINGS.get(validated_task_id)
            if existing is None:
                return False
            if existing.run_id != validated_run_id:
                return False
            mapped_task_id = _TASK_BY_RUN.get(validated_run_id)
            if mapped_task_id != validated_task_id:
                raise ActiveTaskRegistryInvariantError(
                    f"cannot unregister {validated_task_id!r} for {validated_run_id!r}: "
                    f"run index maps to {mapped_task_id!r}"
                )
            del _TASK_BINDINGS[validated_task_id]
            del _TASK_BY_RUN[validated_run_id]
            return True

    @staticmethod
    async def get(task_id: TaskId | str) -> Optional[ActiveTaskBinding]:
        validated_task_id = validate_task_id(task_id)
        async with _LOCK:
            binding = _TASK_BINDINGS.get(validated_task_id)
            if binding is not None:
                _assert_task_binding_index_consistent(binding)
            return binding

    @staticmethod
    async def list_ids() -> list[str]:
        async with _LOCK:
            return [str(task_id) for task_id in _TASK_BINDINGS]

    @staticmethod
    def peek_task_id_for_run(run_id: RunId) -> TaskId | None:
        """Process-local lookup of the active task owning ``run_id``."""
        validated_run_id = validate_run_id(run_id)
        task_id = _TASK_BY_RUN.get(validated_run_id)
        if task_id is None:
            return None
        _assert_run_index_consistent(validated_run_id, task_id)
        return task_id

    @staticmethod
    def clear_for_tests() -> None:
        _TASK_BINDINGS.clear()
        _TASK_BY_RUN.clear()


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
    "ActiveRunOwnershipConflict",
    "ActiveTaskBinding",
    "ActiveTaskOwnershipConflict",
    "ActiveTaskRegistry",
    "ActiveTaskRegistryInvariantError",
    "ActiveTaskRegistryTaskScopeResolver",
]
