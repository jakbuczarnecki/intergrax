# © Artur Czarnecki. All rights reserved.

"""EE-B2 — capacity saturation under fault (EE-B1.2 admission)."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityExceededError,
    ExecutionCapacityOverloadMode,
    ExecutionCapacityPolicy,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.capacity import LocalExecutionCapacityAdmission

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request(tenant: str = "tenant-a") -> ExecutionCapacityAdmissionRequest:
    return ExecutionCapacityAdmissionRequest(
        tenant_id=tenant,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


@pytest.mark.asyncio
async def test_ee_b2_capacity_two_slots_third_rejected() -> None:
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=2,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    admission = LocalExecutionCapacityAdmission(policy)
    p1 = await admission.acquire(_request())
    p2 = await admission.acquire(_request())
    with pytest.raises(ExecutionCapacityExceededError):
        await admission.acquire(_request())
    await p1.release()
    await p2.release()


@pytest.mark.asyncio
async def test_ee_b2_capacity_release_after_faulting_delegate() -> None:
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=1,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    admission = LocalExecutionCapacityAdmission(policy)
    permit = await admission.acquire(_request())
    try:
        raise RuntimeError("delegate_fault")
    except RuntimeError:
        await permit.release()
    second = await admission.acquire(_request())
    await second.release()
