# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Nexus-backed host task terminal publication adapter (public seam)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.runtime.execution.host_task_terminal_publisher import (
    HostTaskTerminalPublisher,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class _NexusHostTaskTerminalPublisher:
    _nexus_loop: NexusLoop

    async def publish_terminal(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> None:
        await self._nexus_loop.publish_host_task_terminal_runtime(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )


def build_nexus_host_task_terminal_publisher(
    nexus_loop: NexusLoop,
) -> HostTaskTerminalPublisher:
    """Composition adapter: host-task terminal publication (operational + canonical terminal)."""
    return _NexusHostTaskTerminalPublisher(_nexus_loop=nexus_loop)


@dataclass(frozen=True, slots=True)
class _NexusRootOrchestrationTerminalPublisher:
    _nexus_loop: NexusLoop

    async def publish_terminal(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> None:
        del run_id, attempt_id, execution_id
        await self._nexus_loop.publish_orchestration_root_terminal_runtime(task)


def build_nexus_root_orchestration_terminal_publisher(
    nexus_loop: NexusLoop,
) -> HostTaskTerminalPublisher:
    """Composition adapter: root orchestration terminal (canonical terminal only)."""
    return _NexusRootOrchestrationTerminalPublisher(_nexus_loop=nexus_loop)


__all__ = [
    "build_nexus_host_task_terminal_publisher",
    "build_nexus_root_orchestration_terminal_publisher",
]
