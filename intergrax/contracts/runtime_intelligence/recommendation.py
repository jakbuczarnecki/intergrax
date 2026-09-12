# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Advisory recommendations — no execution hooks (W6-B)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IntelligenceRecommendationKind(StrEnum):
    """Semantic advisory families — governance and ports decide whether to act."""

    EXECUTION_INSIGHT = "execution_insight"
    RUNTIME_DIAGNOSTIC_HINT = "runtime_diagnostic_hint"
    DECISION_AUDIT_NOTE = "decision_audit_note"
    ADAPTIVE_POLICY_HINT = "adaptive_policy_hint"


@dataclass(frozen=True, slots=True)
class IntelligenceRecommendation:
    """
    Advisory output only.

    Must not encode retries, cancellations, checkpoint mutations, or governance bypass.
    """

    recommendation_id: str
    kind: IntelligenceRecommendationKind
    summary: str
    rationale: str = ""
    priority_label: str = "NORMAL"

    def __post_init__(self) -> None:
        if not self.recommendation_id.strip():
            raise ValueError("recommendation_id must be non-empty")
        if not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.priority_label.strip():
            raise ValueError("priority_label must be non-empty")


__all__ = [
    "IntelligenceRecommendation",
    "IntelligenceRecommendationKind",
]
