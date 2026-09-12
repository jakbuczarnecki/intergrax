# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive action admission boundary — decides; never executes (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.preventive.actions.proposal import PreventiveActionProposal


class PreventiveActionAdmissionVerdict(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


@dataclass(frozen=True, slots=True)
class PreventiveActionAdmissionDecision:
    verdict: PreventiveActionAdmissionVerdict
    reason: str
    approval_id: str | None = None
    decision_id: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")


@dataclass(frozen=True, slots=True)
class PreventiveActionAdmissionContext:
    tenant_id: str
    governance_approved: bool = False
    human_approval_granted: bool = False
    approval_id: str | None = None
    decision_id: str | None = None
    production_target: bool = False


@runtime_checkable
class PreventiveActionAdmissionGate(Protocol):
    """Admission port for preventive proposals — predictive confidence is not authorization."""

    def evaluate(
        self,
        proposal: PreventiveActionProposal,
        context: PreventiveActionAdmissionContext,
    ) -> PreventiveActionAdmissionDecision:
        ...


__all__ = [
    "PreventiveActionAdmissionContext",
    "PreventiveActionAdmissionDecision",
    "PreventiveActionAdmissionGate",
    "PreventiveActionAdmissionVerdict",
]
