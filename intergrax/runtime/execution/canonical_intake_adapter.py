# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical ExecutionRuntime intake adapter (AW-5A seam).

Translates trusted authority admission into ``RootExecutionContext`` and invokes
:class:`ExecutionRuntime` exactly once per dispatch call.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.contracts.execution_intake import (
    CanonicalExecutionIntakePort,
    CanonicalExecutionIntakeRequest,
    CanonicalExecutionIntakeResult,
)
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionOptions,
    resolve_root_execution_context,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class CanonicalExecutionRuntimeAdapter(
    CanonicalExecutionIntakePort[RequestT, ResultT],
    Generic[RequestT, ResultT],
):
    """Runtime-owned adapter — does not mint trusted authority."""

    __slots__ = ("_runtime",)

    def __init__(self, runtime: ExecutionRuntime[RequestT, ResultT]) -> None:
        self._runtime = runtime

    async def dispatch(
        self,
        request: CanonicalExecutionIntakeRequest[RequestT],
    ) -> CanonicalExecutionIntakeResult[ResultT]:
        options = RootExecutionOptions(
            authority=request.trusted_parent_execution_authority,
            run_id=request.run_id,
            attempt_id=request.attempt_id,
            tenant_id=request.tenant_id,
        )
        root_context = resolve_root_execution_context(options)
        result = await self._runtime.execute(request.payload, root_context)
        return CanonicalExecutionIntakeResult(
            run_id=root_context.run_id,
            attempt_id=root_context.attempt_id,
            execution_id=root_context.execution_id,
            result=result,
        )
