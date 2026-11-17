# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Apply resolved host wiring inside Execution Engine owner zone (Tier-1, neutral port only)."""

from __future__ import annotations

from intergrax.contracts.host_orchestration_wiring_port import (
    HostOrchestrationApplicationWiringPort,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def apply_host_orchestration_application_wiring_bundle(
    backend: NexusLoop,
    wiring: HostOrchestrationApplicationWiringPort,
) -> None:
    """Apply resolved application wiring to the private orchestration backend."""
    if not isinstance(wiring, HostOrchestrationApplicationWiringPort):
        raise TypeError(
            "application wiring must implement HostOrchestrationApplicationWiringPort "
            f"(apply, assert_assembly, apply_runtime_wiring); got {type(wiring).__name__}",
        )
    wiring.apply(backend)


__all__ = ["apply_host_orchestration_application_wiring_bundle"]
