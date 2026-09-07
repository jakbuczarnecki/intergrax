# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral ephemeral capability execution port for AW-7B."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.ephemeral_capability_execution import (
    WorkerEphemeralCapabilityExecutionRequest,
    WorkerEphemeralCapabilityExecutionResult,
)


@runtime_checkable
class WorkerEphemeralCapabilityExecutionPort(Protocol):
    """Execute bounded A1 ephemeral capability generation via provider adapter."""

    def execute(
        self,
        request: WorkerEphemeralCapabilityExecutionRequest,
    ) -> WorkerEphemeralCapabilityExecutionResult:
        """Run provider-owned CodeCraft lifecycle and return typed A1 result."""
        ...
