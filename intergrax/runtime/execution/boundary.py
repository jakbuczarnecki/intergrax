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
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_continuation_store import (
    peek_active_execution_continuation_state_store,
)
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.progress_gate import (
    assert_canonical_execution_may_progress,
    load_pending_for_execution_progress,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
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
    task_id: TaskId | None = None


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
        "_continuation_state_store",
    )

    def __init__(
        self,
        delegate: ExecutionDelegate[RequestT, ResultT],
        *,
        admission_hooks: tuple[ExecutionAdmissionHook[RequestT], ...] = (),
        identity: ExecutionIdentityBinding | None = None,
        authority: ParentExecutionAuthority | None = None,
        effective_delegation: EffectiveDelegationAuthority | None = None,
        continuation_state_store: ExecutionContinuationStateStore | None = None,
    ) -> None:
        self._delegate = delegate
        self._admission_hooks = admission_hooks
        self._identity = identity
        self._authority = authority
        self._effective_delegation = effective_delegation
        self._continuation_state_store = continuation_state_store

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

    def _resolve_continuation_state_store(self) -> ExecutionContinuationStateStore | None:
        if self._continuation_state_store is not None:
            return self._continuation_state_store
        return peek_active_execution_continuation_state_store()

    def _continuation_identity_for_progress_gate(self) -> ExecutionContinuationIdentity | None:
        if self._identity is None:
            return None
        task_id = self._identity.task_id
        if task_id is None:
            task_id = ActiveTaskRegistry.peek_task_id_for_run(self._identity.run_id)
        if task_id is None:
            return None
        return ExecutionContinuationIdentity(
            task_id=task_id,
            run_id=self._identity.run_id,
            attempt_id=self._identity.attempt_id,
            execution_id=self._identity.execution_id,
        )

    def _record_safe_pause_after_quiescence(
        self,
        *,
        store: ExecutionContinuationStateStore,
        continuation_identity: ExecutionContinuationIdentity,
    ) -> None:
        pending = load_pending_for_execution_progress(
            store=store,
            identity=continuation_identity,
        )
        if pending is None:
            return
        if pending.lifecycle_state is not ExecutionContinuationLifecycleState.PAUSE_REQUESTED:
            return
        driver = ExecutionContinuationLifecycleDriver(
            ExecutionContinuationService(store),
        )
        driver.record_execution_reached_safe_pause(
            pending.continuation_id,
            execution_pause_established=True,
        )

    async def _run_admission_and_delegate(self, request: RequestT) -> ResultT:
        store = self._resolve_continuation_state_store()
        continuation_identity = (
            self._continuation_identity_for_progress_gate()
            if store is not None
            else None
        )
        if store is not None and continuation_identity is not None:
            assert_canonical_execution_may_progress(
                store=store,
                identity=continuation_identity,
            )

        for hook in self._admission_hooks:
            await hook.admit(request)

        result = await self._delegate.execute(request)

        if store is not None and continuation_identity is not None:
            self._record_safe_pause_after_quiescence(
                store=store,
                continuation_identity=continuation_identity,
            )

        return result
