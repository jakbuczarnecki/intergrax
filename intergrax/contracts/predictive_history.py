# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Prediction history contracts — prediction lifecycle only, not incident truth (PREDICTIVE R2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from intergrax.contracts.predictive_risk import PredictiveWindow

PREDICTIVE_HISTORY_SCHEMA_VERSION: Final = "predictive_risk_history.v1"


class PredictiveHistoryOutcomeStatus(StrEnum):
    """
    Prediction outcome lifecycle — not Problem status.

    Flow: CREATED → OBSERVED → (CONFIRMED | FALSE_POSITIVE) → EVALUATED.
    Terminal evaluation values: CONFIRMED, FALSE_POSITIVE, EXPIRED, UNKNOWN,
    INSUFFICIENT_EVIDENCE.
    """

    CREATED = "created"
    OBSERVED = "observed"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    EVALUATED = "evaluated"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


_TERMINAL_OUTCOMES: frozenset[PredictiveHistoryOutcomeStatus] = frozenset(
    {
        PredictiveHistoryOutcomeStatus.CONFIRMED,
        PredictiveHistoryOutcomeStatus.FALSE_POSITIVE,
        PredictiveHistoryOutcomeStatus.EXPIRED,
        PredictiveHistoryOutcomeStatus.UNKNOWN,
        PredictiveHistoryOutcomeStatus.INSUFFICIENT_EVIDENCE,
    }
)


def is_terminal_prediction_outcome(
    status: PredictiveHistoryOutcomeStatus,
) -> bool:
    return status in _TERMINAL_OUTCOMES


def validate_outcome_lifecycle_transition(
    previous: PredictiveHistoryOutcomeStatus,
    new: PredictiveHistoryOutcomeStatus,
) -> None:
    allowed: dict[PredictiveHistoryOutcomeStatus, frozenset[PredictiveHistoryOutcomeStatus]] = {
        PredictiveHistoryOutcomeStatus.CREATED: frozenset(
            {PredictiveHistoryOutcomeStatus.OBSERVED},
        ),
        PredictiveHistoryOutcomeStatus.OBSERVED: frozenset(
            {
                PredictiveHistoryOutcomeStatus.CONFIRMED,
                PredictiveHistoryOutcomeStatus.FALSE_POSITIVE,
                PredictiveHistoryOutcomeStatus.EXPIRED,
                PredictiveHistoryOutcomeStatus.UNKNOWN,
                PredictiveHistoryOutcomeStatus.INSUFFICIENT_EVIDENCE,
            },
        ),
        PredictiveHistoryOutcomeStatus.CONFIRMED: frozenset(
            {PredictiveHistoryOutcomeStatus.EVALUATED},
        ),
        PredictiveHistoryOutcomeStatus.FALSE_POSITIVE: frozenset(
            {PredictiveHistoryOutcomeStatus.EVALUATED},
        ),
        PredictiveHistoryOutcomeStatus.EXPIRED: frozenset(
            {PredictiveHistoryOutcomeStatus.EVALUATED},
        ),
        PredictiveHistoryOutcomeStatus.UNKNOWN: frozenset(
            {PredictiveHistoryOutcomeStatus.EVALUATED},
        ),
        PredictiveHistoryOutcomeStatus.INSUFFICIENT_EVIDENCE: frozenset(
            {PredictiveHistoryOutcomeStatus.EVALUATED},
        ),
        PredictiveHistoryOutcomeStatus.EVALUATED: frozenset(),
    }
    if new not in allowed.get(previous, frozenset()):
        raise ValueError(
            f"invalid prediction history outcome transition: {previous.value} -> {new.value}",
        )


@dataclass(frozen=True, slots=True)
class PredictiveRiskHistoryRecord:
    """
    Auditable prediction memory — stores prediction lifecycle, not incident truth.

    Forbidden semantics: problem_id as authority, root_cause_claim, diagnostic verdict.
    """

    prediction_run_id: str
    risk_signal_id: str
    tenant_id: str
    analyzer_id: str
    analyzer_version: str
    generated_at: datetime
    prediction_window: PredictiveWindow
    confidence: float
    evidence_refs: tuple[str, ...]
    outcome_status: PredictiveHistoryOutcomeStatus
    subject_identity: str
    risk_type: str
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.prediction_run_id.strip():
            raise ValueError("prediction_run_id must be non-empty")
        if not self.risk_signal_id.strip():
            raise ValueError("risk_signal_id must be non-empty")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if not self.analyzer_version.strip():
            raise ValueError("analyzer_version must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must be non-empty")
        if not self.subject_identity.strip():
            raise ValueError("subject_identity must be non-empty")
        if not self.risk_type.strip():
            raise ValueError("risk_type must be non-empty")


__all__ = [
    "PREDICTIVE_HISTORY_SCHEMA_VERSION",
    "PredictiveHistoryOutcomeStatus",
    "PredictiveRiskHistoryRecord",
    "is_terminal_prediction_outcome",
    "validate_outcome_lifecycle_transition",
]
