# © Artur Czarnecki. All rights reserved.

"""TaskDefinition — platform background task registration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class TaskHandler(Protocol):
    """Developer handler resolved by WorkerRuntime for a registered task_name."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Declarative registration of a background task type."""

    task_name: str
    payload_schema: type[Any]
    handler: TaskHandler
    provider: str = "platform"
