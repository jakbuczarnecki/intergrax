# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Callable, Dict


class TaskExecutionRegistry:
    """
    Registry for logical task handlers executed by worker.

    Maps logical task names to callable handlers.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[..., bytes]] = {}

    def register(
        self,
        task_name: str,
        handler: Callable[..., bytes],
    ) -> None:
        if task_name in self._handlers:
            raise ValueError(
                f"Task '{task_name}' is already registered."
            )

        self._handlers[task_name] = handler

    def get_handler(
        self,
        task_name: str,
    ) -> Callable[..., bytes]:
        if task_name not in self._handlers:
            raise ValueError(
                f"Task '{task_name}' is not registered."
            )

        return self._handlers[task_name]