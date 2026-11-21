# © Artur Czarnecki. All rights reserved.

"""Neutral port for Tier-3 application wiring applied by the Execution Engine (Tier-1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
    HostOrchestrationAssemblyInspectionTarget,
)
from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationRuntimeEventPort,
)


@runtime_checkable
class HostOrchestrationApplicationWiringPort(Protocol):
    """Resolved application wiring that Tier-1 applies to the private orchestration host."""

    def apply(self, target: HostOrchestrationApplicationWiringTarget) -> None:
        """Attach security, guardrail and decision wiring to the host target."""
        ...

    def assert_assembly(self, target: HostOrchestrationAssemblyInspectionTarget) -> None:
        """Raise when the materialized host does not satisfy the application assembly."""
        ...

    def apply_runtime_wiring(
        self,
        target: HostOrchestrationApplicationWiringTarget,
        *,
        event_bus: HostOrchestrationRuntimeEventPort,
    ) -> None:
        """Subscribe observability and attach reliability governance after assembly."""
        ...


__all__ = ["HostOrchestrationApplicationWiringPort"]
