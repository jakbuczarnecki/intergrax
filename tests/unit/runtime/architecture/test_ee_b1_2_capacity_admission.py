# © Artur Czarnecki. All rights reserved.

"""EE-B1.2 — admission port integration and overload semantics."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_capacity import (
    ExecutionCapacityAdmissionDecision,
    ExecutionCapacityAssessmentContext,
    assess_root_execution_capacity,
)
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityAdmissionTimeoutError,
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


def _request() -> ExecutionCapacityAdmissionRequest:
    return ExecutionCapacityAdmissionRequest(
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


@pytest.mark.asyncio
async def test_ee_b1_2_reject_maps_to_exceeded_error() -> None:
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=1,
        overload_mode=ExecutionCapacityOverloadMode.REJECT,
    )
    admission = LocalExecutionCapacityAdmission(policy)
    permit = await admission.acquire(_request())
    preview = assess_root_execution_capacity(
        ExecutionCapacityAssessmentContext(
            active_root_executions=1,
            capacity_limit=1,
            overload_mode=ExecutionCapacityOverloadMode.REJECT,
        ),
    )
    assert preview is ExecutionCapacityAdmissionDecision.REJECT
    with pytest.raises(ExecutionCapacityExceededError):
        await admission.acquire(_request())
    await permit.release()


@pytest.mark.asyncio
async def test_ee_b1_2_timeout_maps_to_defer_semantics() -> None:
    policy = ExecutionCapacityPolicy(
        max_concurrent_root_executions=1,
        overload_mode=ExecutionCapacityOverloadMode.WAIT_WITH_TIMEOUT,
        wait_timeout_seconds=0.02,
    )
    admission = LocalExecutionCapacityAdmission(policy)
    permit = await admission.acquire(_request())
    with pytest.raises(ExecutionCapacityAdmissionTimeoutError):
        await admission.acquire(_request())
    await permit.release()


@pytest.mark.asyncio
async def test_ee_b1_2_acquire_does_not_block_forever_on_reject() -> None:
    policy = ExecutionCapacityPolicy(max_concurrent_root_executions=1)
    admission = LocalExecutionCapacityAdmission(policy)
    permit = await admission.acquire(_request())
    task = asyncio.create_task(admission.acquire(_request()))
    await asyncio.sleep(0.05)
    assert task.done()
    with pytest.raises(ExecutionCapacityExceededError):
        await task
    await permit.release()
