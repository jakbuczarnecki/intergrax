# © Artur Czarnecki. All rights reserved.

"""Default and deterministic suspended-work crash injection adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.execution.crash_injection import (
    ExecutionSuspendedWorkReentryCrashCheckpoint,
    ExecutionSuspendedWorkReentryCrashInjectionPort,
    SimulatedHostProcessLostError,
)


class NoOpExecutionSuspendedWorkReentryCrashInjection(
    ExecutionSuspendedWorkReentryCrashInjectionPort,
):
    """Production default — never aborts."""

    def raise_if_scheduled(
        self,
        checkpoint: ExecutionSuspendedWorkReentryCrashCheckpoint,
    ) -> None:
        del checkpoint


@dataclass
class DeterministicExecutionSuspendedWorkReentryCrashInjection(
    ExecutionSuspendedWorkReentryCrashInjectionPort,
):
    """Single-shot scheduled crash for contract tests."""

    scheduled: ExecutionSuspendedWorkReentryCrashCheckpoint | None = None
    fired: bool = field(default=False, init=False)

    def raise_if_scheduled(
        self,
        checkpoint: ExecutionSuspendedWorkReentryCrashCheckpoint,
    ) -> None:
        if self.fired or self.scheduled is None:
            return
        if checkpoint != self.scheduled:
            return
        self.fired = True
        raise SimulatedHostProcessLostError(checkpoint.value)


__all__ = [
    "DeterministicExecutionSuspendedWorkReentryCrashInjection",
    "NoOpExecutionSuspendedWorkReentryCrashInjection",
]
