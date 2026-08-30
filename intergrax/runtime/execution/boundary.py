# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution boundary coordination skeleton (UE-1B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from intergrax.contracts.delegation_authority import (
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class ExecutionDelegate(Protocol[RequestT, ResultT]):
    """Typed executor invoked exactly once by :class:`ExecutionBoundary`."""

    async def execute(self, request: RequestT) -> ResultT:
        ...


class ExecutionAdmissionHook(Protocol[RequestT]):
    """Admission gate evaluated before delegate execution."""

    async def admit(self, request: RequestT) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ExecutionIdentityBinding:
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    parent_execution_id: ExecutionId | None = None


class ExecutionBoundary(Generic[RequestT, ResultT]):
    """
    Canonical coordination boundary skeleton (UE-1B).

    Receives a typed request, invokes a typed delegate once, and returns the
    typed result unchanged. Does not own subsystem semantics (policy, budget,
    observability, checkpoint, strategy selection, or executor internals).
    Strategy-neutral. Production routing is unchanged at UE-1B.
    """

    __slots__ = (
        "_delegate",
        "_admission_hooks",
        "_identity",
        "_authority",
        "_effective_delegation",
    )

    def __init__(
        self,
        delegate: ExecutionDelegate[RequestT, ResultT],
        *,
        admission_hooks: tuple[ExecutionAdmissionHook[RequestT], ...] = (),
        identity: ExecutionIdentityBinding | None = None,
        authority: ParentExecutionAuthority | None = None,
        effective_delegation: EffectiveDelegationAuthority | None = None,
    ) -> None:
        self._delegate = delegate
        self._admission_hooks = admission_hooks
        self._identity = identity
        self._authority = authority
        self._effective_delegation = effective_delegation

    async def execute(self, request: RequestT) -> ResultT:
        if self._identity is None and self._authority is None:
            return await self._execute_without_context(request)

        identity_token = None
        authority_token = None
        if self._identity is not None:
            identity_token = bind_active_execution_identity(
                run_id=self._identity.run_id,
                attempt_id=self._identity.attempt_id,
                execution_id=self._identity.execution_id,
                parent_execution_id=self._identity.parent_execution_id,
            )
        if self._authority is not None:
            authority_token = bind_active_execution_authority(
                self._authority,
                effective_delegation=self._effective_delegation,
            )
        try:
            return await self._run_admission_and_delegate(request)
        finally:
            if authority_token is not None:
                reset_active_execution_authority(authority_token)
            if identity_token is not None:
                reset_active_execution_identity(identity_token)

    async def _execute_without_context(self, request: RequestT) -> ResultT:
        return await self._run_admission_and_delegate(request)

    async def _run_admission_and_delegate(self, request: RequestT) -> ResultT:
        for hook in self._admission_hooks:
            await hook.admit(request)

        return await self._delegate.execute(request)
