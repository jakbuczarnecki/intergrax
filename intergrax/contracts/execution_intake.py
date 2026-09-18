# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution intake contracts (runtime-owned, provider-neutral)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class CanonicalExecutionInvocationFailed(Exception):
    """Canonical execution started but delegate invocation failed."""

    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    cause: BaseException | None = None

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        validate_attempt_id(self.attempt_id)
        validate_execution_id(self.execution_id)

    def __str__(self) -> str:
        return (
            f"canonical execution invocation failed "
            f"(run={self.run_id}, attempt={self.attempt_id}, execution={self.execution_id})"
        )


@dataclass(frozen=True, slots=True)
class CanonicalExecutionIntakeRequest(Generic[PayloadT]):
    """Provider-neutral runtime intake request after trusted authority admission."""

    payload: PayloadT
    trusted_parent_execution_authority: ParentExecutionAuthority
    admitted_governance_identity: AdmittedRootGovernanceIdentity
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    task_id: TaskId | None = None
    segment_predecessor_root_execution_id: ExecutionId | None = None

    @property
    def tenant_id(self) -> str:
        return self.admitted_governance_identity.tenant_id

    @property
    def workspace_id(self) -> str:
        return self.admitted_governance_identity.workspace_id

    @property
    def principal_id(self) -> str:
        return self.admitted_governance_identity.principal_id

    def __post_init__(self) -> None:
        if type(self.trusted_parent_execution_authority) is not ParentExecutionAuthority:
            raise TypeError(
                "trusted_parent_execution_authority must be ParentExecutionAuthority"
            )
        if type(self.admitted_governance_identity) is not AdmittedRootGovernanceIdentity:
            raise TypeError("admitted_governance_identity must be AdmittedRootGovernanceIdentity")
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if self.task_id is not None:
            validate_task_id(self.task_id)
        if self.segment_predecessor_root_execution_id is not None:
            validate_execution_id(self.segment_predecessor_root_execution_id)


@dataclass(frozen=True, slots=True)
class CanonicalExecutionIntakeResult(Generic[ResultT]):
    """Canonical runtime intake result with platform-owned execution identities."""

    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    result: ResultT

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        validate_attempt_id(self.attempt_id)
        validate_execution_id(self.execution_id)


@runtime_checkable
class CanonicalExecutionIntakePort(Protocol[PayloadT, ResultT]):
    """Smallest stable runtime port for canonical execution dispatch."""

    async def dispatch(
        self,
        request: CanonicalExecutionIntakeRequest[PayloadT],
    ) -> CanonicalExecutionIntakeResult[ResultT]:
        """Dispatch one execution through canonical ExecutionRuntime."""
        ...
