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
from intergrax.runtime.execution.authority.policy import (
    ChildAuthorityContext,
    DefaultStrictAuthorityPolicy,
    ExecutionAuthorityPolicy,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionDelegate,
    ExecutionIdentityBinding,
)
from intergrax.runtime.governance.active_execution_authority import (
    require_active_execution_authority,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class ChildExecutionRunner(Generic[RequestT, ResultT]):
    """
    Mint a child ExecutionId under the active parent Execution and route work
    through :class:`ExecutionBoundary`.
    """

    __slots__ = ("_authority_policy",)

    def __init__(
        self,
        authority_policy: ExecutionAuthorityPolicy | None = None,
    ) -> None:
        self._authority_policy = (
            authority_policy
            if authority_policy is not None
            else DefaultStrictAuthorityPolicy()
        )

    async def execute(
        self,
        *,
        request: RequestT,
        delegate: ExecutionDelegate[RequestT, ResultT],
        admission_hooks: tuple[ExecutionAdmissionHook[RequestT], ...] = (),
        requested_permission_scopes: tuple[str, ...] | None = None,
    ) -> ResultT:
        parent_run_id, parent_attempt_id = require_active_execution_identity()
        parent_execution_id = require_active_execution_id()
        parent_authority = require_active_execution_authority()

        resolution = self._authority_policy.resolve_child_authority(
            ChildAuthorityContext(
                parent_authority=parent_authority,
                requested_permission_scopes=requested_permission_scopes,
            )
        )
        child_authority = resolution.authority
        effective = resolution.effective_delegation

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
            authority=child_authority,
            effective_delegation=effective,
        )
        return await boundary.execute(request)
