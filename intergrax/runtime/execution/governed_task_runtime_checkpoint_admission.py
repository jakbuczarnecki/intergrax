# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Materialize canonical task RuntimeCheckpoint at governed root admission (UCA-6C-R6-R5.9-R1-R1-R1)."""

from __future__ import annotations

from intergrax.contracts.execution.qualified_capability_execution_intake import (
    QualifiedCapabilityExecutionIntakePayload,
)
from intergrax.contracts.execution_identity import validate_task_id
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)
from intergrax.runtime.long_running.checkpoint_builder import (
    materialize_task_runtime_checkpoint_for_active_execution,
)


class QualifiedCapabilityRuntimeCheckpointMaterializationAdmission:
    """ExecutionRuntime admission hook — EE-owned runtime checkpoint before delegate work."""

    async def admit(
        self,
        request: QualifiedCapabilityExecutionIntakePayload,
    ) -> None:
        task = peek_governed_execution_task()
        if task is None:
            return
        expected_task_id = validate_task_id(request.task_id)
        if task.task_id != expected_task_id:
            raise ValueError(
                "governed execution task_id mismatch at runtime checkpoint admission: "
                f"{task.task_id!r} != {expected_task_id!r}",
            )
        materialize_task_runtime_checkpoint_for_active_execution(task)


__all__ = ["QualifiedCapabilityRuntimeCheckpointMaterializationAdmission"]
