# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Port between ERL recovery recommendations and Unified Execution Runtime lifecycle authority."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.recovery_decision import RecoveryDecision


class RecoveryLifecycleIntent(BaseModel):
    """Handoff artifact — ERL produces; execution runtime consumes and mutates lifecycle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    decision: RecoveryDecision


@runtime_checkable
class ExecutionLifecyclePort(Protocol):
    """
    Execution runtime port — owns pause, resume, terminate, and HITL routing.

    ERL recovery orchestration must not implement or invoke this port; adapters wire UER.
    """

    def apply_recovery_lifecycle_intent(self, intent: RecoveryLifecycleIntent) -> None:
        """Apply a platform recovery recommendation to execution lifecycle state."""


__all__ = [
    "ExecutionLifecyclePort",
    "RecoveryLifecycleIntent",
]
