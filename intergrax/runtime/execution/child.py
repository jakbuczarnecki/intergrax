# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical child execution lineage (UE-7A)."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.contracts.execution_identity import (
    mint_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionDelegate,
    ExecutionIdentityBinding,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class ChildExecutionRunner(Generic[RequestT, ResultT]):
    """
    Mint a child ExecutionId under the active parent Execution and route work
    through :class:`ExecutionBoundary`.
    """

    __slots__ = ()

    async def execute(
        self,
        *,
        request: RequestT,
        delegate: ExecutionDelegate[RequestT, ResultT],
        admission_hooks: tuple[ExecutionAdmissionHook[RequestT], ...] = (),
    ) -> ResultT:
        parent_run_id, parent_attempt_id = require_active_execution_identity()
        parent_execution_id = require_active_execution_id()

        child_execution_id = mint_execution_id()
        identity = ExecutionIdentityBinding(
            run_id=parent_run_id,
            attempt_id=parent_attempt_id,
            execution_id=child_execution_id,
            parent_execution_id=parent_execution_id,
        )
        boundary = ExecutionBoundary[RequestT, ResultT](
            delegate,
            admission_hooks=admission_hooks,
            identity=identity,
        )
        return await boundary.execute(request)
