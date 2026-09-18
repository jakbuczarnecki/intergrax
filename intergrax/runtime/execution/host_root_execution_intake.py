# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host root intake bridge — canonical intake to Execution facade (GR-2-R3 internal)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.execution_capacity_admission import ExecutionCapacityPermit
from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionIntakeResult,
)
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionOptions,
    resolve_root_execution_context,
)
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.task.task import TaskResult

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class HostRootExecutionIntakePayload(Generic[RequestT]):
    execution_request: RequestT
    held_root_capacity_permit: ExecutionCapacityPermit | None = None


class HostFacadeRootExecutionIntake(
    CanonicalExecutionIntakePort[
        HostRootExecutionIntakePayload[ExecutionRequest[TaskExecutionInput, TaskResult]],
        TaskResult,
    ],
):
    """Internal host bridge — ExecutionRuntime supplied per task at composition."""

    __slots__ = ("_runtime",)

    def __init__(
        self,
        runtime: ExecutionRuntime[
            ExecutionRequest[TaskExecutionInput, TaskResult],
            TaskResult,
        ],
    ) -> None:
        self._runtime = runtime

    async def dispatch(
        self,
        request: CanonicalExecutionIntakeRequest[
            HostRootExecutionIntakePayload[ExecutionRequest[TaskExecutionInput, TaskResult]]
        ],
    ) -> CanonicalExecutionIntakeResult[TaskResult]:
        payload = request.payload
        options = RootExecutionOptions(
            authority=request.trusted_parent_execution_authority,
            governance_identity=request.admitted_governance_identity,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            execution_id=request.execution_id,
            task_id=request.task_id,
            segment_predecessor_root_execution_id=request.segment_predecessor_root_execution_id,
        )
        root_context = resolve_root_execution_context(options)
        execution = Execution(self._runtime)
        result = await execution.execute(
            payload.execution_request,
            options=options,
            held_root_capacity_permit=payload.held_root_capacity_permit,
        )
        return CanonicalExecutionIntakeResult(
            run_id=root_context.run_id,
            attempt_id=root_context.attempt_id,
            execution_id=root_context.execution_id,
            result=result,
        )
