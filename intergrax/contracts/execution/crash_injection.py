# © Artur Czarnecki. All rights reserved.

"""Typed crash-injection checkpoints for execution recovery proofs (UCA-6C-R6-R5.9-R3)."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class ExecutionSuspendedWorkReentryCrashCheckpoint(str, Enum):
    """Semantic checkpoints on the canonical suspended-work re-entry path."""

    AFTER_CLAIM_BEFORE_TOOL_RUNTIME = "after_claim_before_tool_runtime"
    AFTER_EFFECT_COMMIT_BEFORE_CONSUME = "after_effect_commit_before_consume"
    AFTER_CONSUME_BEFORE_TERMINAL = "after_consume_before_terminal"
    AFTER_TERMINAL_BEFORE_RETURN = "after_terminal_before_return"


class ToolRuntimeEffectCrashCheckpoint(str, Enum):
    """Semantic checkpoints on RuntimeToolInvoker effect/idempotency boundary."""

    AFTER_TOOL_RUNTIME_ADMISSION_BEFORE_BACKEND = (
        "after_tool_runtime_admission_before_backend"
    )
    AFTER_BACKEND_BEFORE_EFFECT_COMMIT = "after_backend_before_effect_commit"


class SimulatedHostProcessLostError(RuntimeError):
    """Abort the current host without implying business-level failure semantics."""

    def __init__(self, checkpoint: str) -> None:
        super().__init__(f"simulated host process lost at {checkpoint}")
        self.checkpoint = checkpoint


class ExecutionSuspendedWorkReentryCrashInjectionPort(Protocol):
    """Optional observer that may abort the host at a re-entry checkpoint."""

    def raise_if_scheduled(
        self,
        checkpoint: ExecutionSuspendedWorkReentryCrashCheckpoint,
    ) -> None:
        """Raise SimulatedHostProcessLostError when this checkpoint is scheduled."""


class ToolRuntimeEffectCrashInjectionPort(Protocol):
    """Optional observer that may abort the host at a ToolRuntime effect checkpoint."""

    def raise_if_scheduled(
        self,
        checkpoint: ToolRuntimeEffectCrashCheckpoint,
    ) -> None:
        """Raise SimulatedHostProcessLostError when this checkpoint is scheduled."""


__all__ = [
    "ExecutionSuspendedWorkReentryCrashCheckpoint",
    "ExecutionSuspendedWorkReentryCrashInjectionPort",
    "SimulatedHostProcessLostError",
    "ToolRuntimeEffectCrashCheckpoint",
    "ToolRuntimeEffectCrashInjectionPort",
]
