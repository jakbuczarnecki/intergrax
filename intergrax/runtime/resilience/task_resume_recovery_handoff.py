# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""TASK_RESUME recovery start handoff (W3-C §3.3).

Orders recovery admission before root execution capacity admission and releases the
recovery-start permit only after execution-width ownership is established.
"""

from __future__ import annotations

from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionPort,
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityPermit,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.recovery_admission import (
    RecoveryAdmissionPort,
    RecoveryAdmissionRequest,
    RecoveryKind,
)


async def handoff_task_resume_recovery_start(
    *,
    tenant_id: str | None,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    recovery_admission: RecoveryAdmissionPort | None,
    execution_capacity_admission: ExecutionCapacityAdmissionPort | None,
) -> ExecutionCapacityPermit | None:
    """Acquire recovery start + optional root capacity; release recovery after W1 handoff."""
    recovery_permit = None
    capacity_permit: ExecutionCapacityPermit | None = None
    if recovery_admission is not None:
        recovery_permit = await recovery_admission.acquire(
            RecoveryAdmissionRequest(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                recovery_kind=RecoveryKind.TASK_RESUME,
            ),
        )
    try:
        if execution_capacity_admission is not None:
            capacity_permit = await execution_capacity_admission.acquire(
                ExecutionCapacityAdmissionRequest(
                    tenant_id=tenant_id,
                    task_id=task_id,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=execution_id,
                ),
            )
        if recovery_permit is not None:
            await recovery_permit.release()
            recovery_permit = None
        return capacity_permit
    except BaseException:
        if recovery_permit is not None:
            await recovery_permit.release()
        if capacity_permit is not None:
            await capacity_permit.release()
        raise
