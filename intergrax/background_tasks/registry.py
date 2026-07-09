# © Artur Czarnecki. All rights reserved.

"""TaskRegistry — platform catalog of registered background tasks."""

from __future__ import annotations

from intergrax.background_tasks.definition import TaskDefinition
from intergrax.queueing.worker.registry import TaskExecutionRegistry


class UnknownTaskError(KeyError):
    """Raised when task_name is not registered."""


class TaskRegistry:
    """Maps task_name -> TaskDefinition and exposes TaskExecutionRegistry handlers."""

    def __init__(self) -> None:
        self._definitions: dict[str, TaskDefinition] = {}

    def register(self, definition: TaskDefinition) -> None:
        if not definition.task_name.strip():
            raise ValueError("task_name must not be blank")
        self._definitions[definition.task_name] = definition

    def resolve(self, task_name: str) -> TaskDefinition:
        try:
            return self._definitions[task_name]
        except KeyError as exc:
            raise UnknownTaskError(f"unknown_task_name:{task_name}") from exc

    def has_task(self, task_name: str) -> bool:
        return task_name in self._definitions

    def to_execution_registry(self) -> TaskExecutionRegistry:
        registry = TaskExecutionRegistry()
        for definition in self._definitions.values():
            registry.register(definition.task_name, definition.handler)
        return registry

    def bind_execution_registry(self, registry: TaskExecutionRegistry) -> None:
        for definition in self._definitions.values():
            registry.register(definition.task_name, definition.handler)
