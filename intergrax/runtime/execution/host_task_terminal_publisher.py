# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host task terminal publication port (NPSC-3B-R1)."""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.runtime.task.task import Task


class HostTaskTerminalPublisher(Protocol):
    """Provider-neutral port for terminal host task runtime publication."""

    async def publish_terminal(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> None: ...
