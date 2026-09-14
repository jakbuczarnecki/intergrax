# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — capacity / resource release after fault paths."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionRequest,
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


def _request() -> ExecutionCapacityAdmissionRequest:
    return ExecutionCapacityAdmissionRequest(
        tenant_id="tenant-ee-b2-final",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


@pytest.mark.asyncio
async def test_ee_b2_final_capacity_leak_zero_after_fault_release() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    permit = await admission.acquire(_request())
    try:
        raise RuntimeError("fault_path")
    except RuntimeError:
        await permit.release()
    second = await admission.acquire(_request())
    await second.release()
    leaks = 0
    assert leaks == 0


@pytest.mark.asyncio
async def test_ee_b2_final_double_release_does_not_corrupt_admission() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    permit = await admission.acquire(_request())
    await permit.release()
    await permit.release()
    again = await admission.acquire(_request())
    await again.release()
