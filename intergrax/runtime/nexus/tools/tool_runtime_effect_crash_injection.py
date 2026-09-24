# © Artur Czarnecki. All rights reserved.

"""Default and deterministic ToolRuntime effect crash injection adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.execution.crash_injection import (
    SimulatedHostProcessLostError,
    ToolRuntimeEffectCrashCheckpoint,
    ToolRuntimeEffectCrashInjectionPort,
)


class NoOpToolRuntimeEffectCrashInjection(ToolRuntimeEffectCrashInjectionPort):
    """Production default — never aborts."""

    def raise_if_scheduled(
        self,
        checkpoint: ToolRuntimeEffectCrashCheckpoint,
    ) -> None:
        del checkpoint


@dataclass
class DeterministicToolRuntimeEffectCrashInjection(
    ToolRuntimeEffectCrashInjectionPort,
):
    """Single-shot scheduled crash for contract tests."""

    scheduled: ToolRuntimeEffectCrashCheckpoint | None = None
    fired: bool = field(default=False, init=False)

    def raise_if_scheduled(
        self,
        checkpoint: ToolRuntimeEffectCrashCheckpoint,
    ) -> None:
        if self.fired or self.scheduled is None:
            return
        if checkpoint != self.scheduled:
            return
        self.fired = True
        raise SimulatedHostProcessLostError(checkpoint.value)


__all__ = [
    "DeterministicToolRuntimeEffectCrashInjection",
    "NoOpToolRuntimeEffectCrashInjection",
]
