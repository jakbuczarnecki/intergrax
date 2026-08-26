# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task → canonical ExecutionRequest projection (UE-2B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.task.task import Task

OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class TaskExecutionInput:
    """
    Portable Task work-intent projection for canonical execution requests.

    ``capability`` is an application/task capability label (e.g.
    ``"incident.investigation"``) — not :class:`ExecutionCapability`, which
    describes execution requirements supplied explicitly by the migration caller.
    """

    message: str
    capability: str | None = None
    intent: str | None = None


def execution_request_from_task(
    task: Task,
    *,
    capabilities: frozenset[ExecutionCapability] = frozenset(),
    output_type: type[OutputT] | None = None,
) -> ExecutionRequest[TaskExecutionInput, OutputT]:
    """
    Project portable Task work intent into a canonical :class:`ExecutionRequest`.

    Reads only ``task.message``, ``task.context.capability``, and
    ``task.context.intent``. Does not infer :class:`ExecutionCapability` values,
    copy identity/metadata, or mutate ``task``.
    """
    return ExecutionRequest(
        input=TaskExecutionInput(
            message=task.message,
            capability=task.context.capability,
            intent=task.context.intent,
        ),
        output_type=output_type,
        capabilities=capabilities,
    )
