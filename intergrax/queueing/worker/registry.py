# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Protocol

from pydantic import BaseModel

from intergrax.tools.execution_models import ToolExecutionResult

if TYPE_CHECKING:
    from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity


class BackgroundTaskHandler(Protocol):
    """Canonical handler contract for supported background logical tasks."""

    def __call__(
        self,
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key: Optional[str],
        execution_identity: BackgroundExecutionIdentity,
    ) -> ToolExecutionResult[BaseModel]:
        ...


class TaskExecutionRegistry:
    """
    Registry for logical task handlers executed by worker.

    Maps logical task names to canonical BackgroundTaskHandler callables.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, BackgroundTaskHandler] = {}

    def register(
        self,
        task_name: str,
        handler: BackgroundTaskHandler,
    ) -> None:
        if task_name in self._handlers:
            raise ValueError(
                f"Task '{task_name}' is already registered."
            )

        self._handlers[task_name] = handler

    def get_handler(
        self,
        task_name: str,
    ) -> BackgroundTaskHandler:
        if task_name not in self._handlers:
            raise ValueError(
                f"Task '{task_name}' is not registered."
            )

        return self._handlers[task_name]

    def unregister(self, task_name: str) -> bool:
        """Remove a task handler and report whether it was registered."""
        return self._handlers.pop(task_name, None) is not None
