# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution intake contracts (runtime-owned, provider-neutral)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class CanonicalExecutionIntakeRequest(Generic[PayloadT]):
    """Provider-neutral runtime intake request after trusted authority admission."""

    payload: PayloadT
    trusted_parent_execution_authority: ParentExecutionAuthority
    tenant_id: str
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None

    def __post_init__(self) -> None:
        if type(self.trusted_parent_execution_authority) is not ParentExecutionAuthority:
            raise TypeError(
                "trusted_parent_execution_authority must be ParentExecutionAuthority"
            )
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)


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
