# © Artur Czarnecki. All rights reserved.

"""Pre-effect protected work admission contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable


class ExecutionProtectedWorkAdmissionResult(StrEnum):
    AVAILABLE = "available"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@runtime_checkable
class ExecutionCancellationView(Protocol):
    def is_cancelled(self) -> bool:
        """Whether cooperative cancellation is active for this execution scope."""

    def cancellation_reason(self) -> str | None:
        """Optional operator/user cancellation reason for evidence."""


class ExecutionProtectedWorkAdmissionPort(Protocol):
    def assert_can_start_protected_work(self) -> ExecutionProtectedWorkAdmissionResult:
        """Read-only admission check before a new protected side effect may start."""
