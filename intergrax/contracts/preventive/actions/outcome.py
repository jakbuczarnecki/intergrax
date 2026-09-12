# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive action outcome feedback for prediction quality (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class PreventiveActionObservedOutcome(StrEnum):
    INCIDENT_AVOIDED = "INCIDENT_AVOIDED"
    INEFFECTIVE = "INEFFECTIVE"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class PreventiveActionOutcomeEvaluation:
    proposal_id: str
    tenant_id: str
    analyzer_id: str
    risk_signal_refs: tuple[str, ...]
    observed: PreventiveActionObservedOutcome
    evidence_refs: tuple[str, ...]
    evaluated_at: datetime

    def __post_init__(self) -> None:
        if not self.proposal_id.startswith("pract_prop_"):
            raise ValueError("proposal_id must be pract_prop_*")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")


__all__ = ["PreventiveActionObservedOutcome", "PreventiveActionOutcomeEvaluation"]
