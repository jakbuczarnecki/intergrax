# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""External operation cancellation and status inquiry ports (W4-C).

Scope: request cancellation and status inquiry only — not retry, scheduling,
recovery, or ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ExternalOperationIntentState(StrEnum):
    """Caller/system intent to finish the logical operation."""

    ACTIVE = "ACTIVE"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"


class ExternalOperationPhysicalState(StrEnum):
    """Observed physical execution on an external dependency."""

    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class ExternalOperationState:
    """Combined durable view for one logical operation_id."""

    operation_id: str
    intent_state: ExternalOperationIntentState
    physical_state: ExternalOperationPhysicalState
    created_at: datetime
    updated_at: datetime
    revision: int
    owner_token: str | None = None


class ExternalOperationCancellationError(RuntimeError):
    """Base error for external operation cancellation plane."""


class ExternalOperationNotFoundError(ExternalOperationCancellationError):
    """No durable record for the requested operation_id."""


@runtime_checkable
class ExternalOperationCancellationPort(Protocol):
    """Provider-specific cancellation signal (best-effort)."""

    async def request_cancel(self, operation_id: str) -> None:
        """Ask the external system to stop; does not imply termination."""
        ...


@runtime_checkable
class ExternalOperationStatusPort(Protocol):
    """Optional provider-side status when supported."""

    async def get_status(self, operation_id: str) -> ExternalOperationPhysicalState:
        """Return provider-observed physical state for operation_id."""
        ...
