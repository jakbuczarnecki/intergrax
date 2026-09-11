# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recovery start admission contracts (W3-C).

Controls concurrent recovery **starts** only — not in-flight recovery duration,
execution ownership, retry, or scheduler claim semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_run_id,
    validate_task_id,
)


class RecoveryAdmissionError(RuntimeError):
    """Base error for recovery start admission."""


class RecoveryAdmissionExceededError(RecoveryAdmissionError):
    """No recovery start slot available under REJECT overload policy."""


class RecoveryAdmissionTimeoutError(RecoveryAdmissionError):
    """No recovery start slot became available before WAIT_WITH_TIMEOUT elapsed."""


class RecoveryAdmissionPolicyMissingError(RecoveryAdmissionError):
    """Admission is active but no policy is configured for the recovery kind."""


class RecoveryKind(StrEnum):
    """Typed recovery start domain (not tenant, not root execution)."""

    TASK_RESUME = "TASK_RESUME"
    PARTIAL_TOPOLOGY = "PARTIAL_TOPOLOGY"


class RecoveryAdmissionOverloadMode(StrEnum):
    """Overload behavior when recovery start slots are saturated."""

    REJECT = "REJECT"
    WAIT_WITH_TIMEOUT = "WAIT_WITH_TIMEOUT"


class RecoveryAdmissionPolicy(BaseModel):
    """Explicit recovery start concurrency policy per recovery kind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_concurrent_recovery_starts: int = Field(ge=1)
    overload_mode: RecoveryAdmissionOverloadMode = RecoveryAdmissionOverloadMode.REJECT
    wait_timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_wait_timeout(self) -> RecoveryAdmissionPolicy:
        if self.overload_mode is RecoveryAdmissionOverloadMode.WAIT_WITH_TIMEOUT:
            if self.wait_timeout_seconds is None:
                raise ValueError(
                    "wait_timeout_seconds must be set when overload_mode is "
                    "WAIT_WITH_TIMEOUT"
                )
        elif self.wait_timeout_seconds is not None:
            raise ValueError(
                "wait_timeout_seconds applies only when overload_mode is "
                "WAIT_WITH_TIMEOUT"
            )
        return self


@dataclass(frozen=True, slots=True)
class RecoveryAdmissionRequest:
    """Identity for one recovery start slot (admission metadata only)."""

    tenant_id: str | None
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    recovery_kind: RecoveryKind

    def __post_init__(self) -> None:
        if self.tenant_id is not None and not isinstance(self.tenant_id, str):
            raise TypeError("tenant_id must be str or None")
        object.__setattr__(self, "task_id", validate_task_id(self.task_id))
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        object.__setattr__(self, "attempt_id", validate_attempt_id(self.attempt_id))
        if not isinstance(self.recovery_kind, RecoveryKind):
            raise TypeError("recovery_kind must be RecoveryKind")


@runtime_checkable
class RecoveryAdmissionPermit(Protocol):
    """Held slot for one recovery start; release exactly once."""

    async def release(self) -> None:
        """Return the recovery start slot to the admission pool."""
        ...


@runtime_checkable
class RecoveryAdmissionPort(Protocol):
    """Pluginable recovery start admission (local, distributed, etc.)."""

    async def acquire(
        self,
        request: RecoveryAdmissionRequest,
    ) -> RecoveryAdmissionPermit:
        """Reserve one recovery start slot or fail fast / time out per policy."""
        ...
