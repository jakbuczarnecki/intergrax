# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionResult

if TYPE_CHECKING:
    from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity

BACKGROUND_TASK_DECLARES_EXTERNAL_SIDE_EFFECTS = "__intergrax_background_declares_external_side_effects__"
BACKGROUND_TASK_EXTERNAL_EFFECTS_VIA_RUNTIME_TOOL_INVOKER = (
    "__intergrax_background_external_effects_via_runtime_tool_invoker__"
)


def mark_background_task_declares_external_side_effects(handler: object) -> object:
    setattr(handler, BACKGROUND_TASK_DECLARES_EXTERNAL_SIDE_EFFECTS, True)
    return handler


def mark_background_task_external_effects_via_runtime_tool_invoker(handler: object) -> object:
    setattr(handler, BACKGROUND_TASK_EXTERNAL_EFFECTS_VIA_RUNTIME_TOOL_INVOKER, True)
    return handler


def background_task_handler_declares_external_side_effects(handler: object) -> bool:
    return bool(getattr(handler, BACKGROUND_TASK_DECLARES_EXTERNAL_SIDE_EFFECTS, False))


def assert_background_task_external_effects_use_runtime_tool_invoker(handler: object) -> None:
    if not background_task_handler_declares_external_side_effects(handler):
        return
    if not getattr(handler, BACKGROUND_TASK_EXTERNAL_EFFECTS_VIA_RUNTIME_TOOL_INVOKER, False):
        raise ValueError(
            "Background logical tasks that declare external side effects must route "
            "provider mutations through RuntimeToolInvoker.",
        )


class BackgroundTaskHandler(Protocol):
    """Canonical handler contract for supported background logical tasks."""

    def __call__(
        self,
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: str | None,
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[BaseModel]: ...


class TaskExecutionRegistry:
    """
    Registry for logical task handlers executed by worker.

    Maps logical task names to canonical BackgroundTaskHandler callables.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, BackgroundTaskHandler] = {}

    def register(
        self,
        task_name: str,
        handler: BackgroundTaskHandler,
    ) -> None:
        assert_background_task_external_effects_use_runtime_tool_invoker(handler)
        if task_name in self._handlers:
            raise ValueError(f"Task '{task_name}' is already registered.")

        self._handlers[task_name] = handler

    def get_handler(
        self,
        task_name: str,
    ) -> BackgroundTaskHandler:
        if task_name not in self._handlers:
            raise ValueError(f"Task '{task_name}' is not registered.")

        return self._handlers[task_name]

    def unregister(self, task_name: str) -> bool:
        """Remove a task handler and report whether it was registered."""
        return self._handlers.pop(task_name, None) is not None
