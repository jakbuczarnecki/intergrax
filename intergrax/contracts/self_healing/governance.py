# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing governance admission — decides; never executes (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.decision import SelfHealingDecision


class SelfHealingAdmissionVerdict(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


@dataclass(frozen=True, slots=True)
class SelfHealingAdmissionDecision:
    verdict: SelfHealingAdmissionVerdict
    reason: str
    approval_id: str | None = None
    decision_id: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")


@dataclass(frozen=True, slots=True)
class SelfHealingAdmissionContext:
    tenant_id: str
    governance_approved: bool = False
    human_approval_granted: bool = False
    approval_id: str | None = None
    decision_id: str | None = None
    production_target: bool = False


@runtime_checkable
class SelfHealingAdmissionGate(Protocol):
    """Admission port — strategy confidence is not authorization."""

    def evaluate(
        self,
        decision: SelfHealingDecision,
        context: SelfHealingAdmissionContext,
    ) -> SelfHealingAdmissionDecision:
        ...


__all__ = [
    "SelfHealingAdmissionContext",
    "SelfHealingAdmissionDecision",
    "SelfHealingAdmissionGate",
    "SelfHealingAdmissionVerdict",
]
