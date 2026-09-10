# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Root execution capacity admission contracts (W1-A).

Orthogonal to :class:`ExecutionAdmissionHook` — physical process-local slots only.
Distributed admission is deferred; ports are async for future implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ExecutionCapacityAdmissionError(RuntimeError):
    """Base error for root execution capacity admission."""


class ExecutionCapacityExceededError(ExecutionCapacityAdmissionError):
    """No slot available under REJECT overload policy."""


class ExecutionCapacityAdmissionTimeoutError(ExecutionCapacityAdmissionError):
    """No slot became available before WAIT_WITH_TIMEOUT elapsed."""


class ExecutionCapacityOverloadMode(StrEnum):
    """Overload behavior when root execution slots are saturated."""

    REJECT = "REJECT"
    WAIT_WITH_TIMEOUT = "WAIT_WITH_TIMEOUT"


class ExecutionCapacityPolicy(BaseModel):
    """Process-local root execution concurrency policy (explicit construction required)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_concurrent_root_executions: int = Field(ge=1)
    overload_mode: ExecutionCapacityOverloadMode = ExecutionCapacityOverloadMode.REJECT
    wait_timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_wait_timeout(self) -> ExecutionCapacityPolicy:
        if self.overload_mode is ExecutionCapacityOverloadMode.WAIT_WITH_TIMEOUT:
            if self.wait_timeout_seconds is None:
                raise ValueError(
                    "wait_timeout_seconds must be set when overload_mode is WAIT_WITH_TIMEOUT"
                )
        elif self.wait_timeout_seconds is not None:
            raise ValueError(
                "wait_timeout_seconds applies only when overload_mode is WAIT_WITH_TIMEOUT"
            )
        return self


@dataclass(frozen=True, slots=True)
class ExecutionCapacityAdmissionRequest:
    """Immutable identity for one root execution capacity slot."""

    tenant_id: str | None
    task_id: TaskId | None
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId

    def __post_init__(self) -> None:
        if self.tenant_id is not None and not isinstance(self.tenant_id, str):
            raise TypeError("tenant_id must be str or None")
        object.__setattr__(
            self,
            "task_id",
            None if self.task_id is None else validate_task_id(self.task_id),
        )
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "attempt_id", validate_attempt_id(self.attempt_id))
        object.__setattr__(self, "execution_id", validate_execution_id(self.execution_id))


@runtime_checkable
class ExecutionCapacityPermit(Protocol):
    """Held slot for one root execution; release exactly once (idempotent implementations allowed)."""

    async def release(self) -> None:
        """Return the slot to the admission pool."""
        ...


@runtime_checkable
class ExecutionCapacityAdmissionPort(Protocol):
    """Pluginable root execution capacity admission (local, distributed, etc.)."""

    async def acquire(
        self,
        request: ExecutionCapacityAdmissionRequest,
    ) -> ExecutionCapacityPermit:
        """Reserve one root execution slot or fail fast / time out per policy."""
        ...
