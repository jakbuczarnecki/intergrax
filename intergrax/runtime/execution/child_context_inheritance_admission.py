# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Child context inheritance admission hook (Execution-owned)."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.contracts.child_execution_context_inheritance import (
    ChildExecutionContextInheritancePort,
    ChildExecutionContextInheritanceRequest,
)
from intergrax.contracts.execution_identity import ExecutionId

RequestT = TypeVar("RequestT")


class ChildExecutionContextInheritanceAdmissionHook(Generic[RequestT]):
    """Runs profile/context inheritance after child identity is bound."""

    __slots__ = ("_port", "_request")

    def __init__(
        self,
        port: ChildExecutionContextInheritancePort,
        request: ChildExecutionContextInheritanceRequest,
    ) -> None:
        self._port = port
        self._request = request

    async def admit(self, request: RequestT) -> None:
        del request
        self._port.inherit_child_context(self._request)


def build_child_context_inheritance_admission_hook(
    port: ChildExecutionContextInheritancePort,
    *,
    parent_execution_id: ExecutionId,
    child_execution_id: ExecutionId,
) -> ChildExecutionContextInheritanceAdmissionHook[RequestT]:
    inheritance_request = ChildExecutionContextInheritanceRequest(
        parent_execution_id=parent_execution_id,
        child_execution_id=child_execution_id,
    )
    return ChildExecutionContextInheritanceAdmissionHook(port, inheritance_request)


__all__ = [
    "ChildExecutionContextInheritanceAdmissionHook",
    "build_child_context_inheritance_admission_hook",
]
