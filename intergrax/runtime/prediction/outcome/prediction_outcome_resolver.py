# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Resolve whether past predictions were accurate — never mint Problems (PREDICTIVE R2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
)


@dataclass(frozen=True, slots=True)
class PredictionFutureEvidenceSnapshot:
    """Readonly facts for outcome resolution — not diagnostic authority."""

    observed_at: datetime
    evidence_refs: tuple[str, ...]
    execution_failed: bool
    problem_created_for_subject: bool
    matching_risk_keywords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PredictionOutcomeResolution:
    outcome_status: PredictiveHistoryOutcomeStatus
    rationale: str


@dataclass(frozen=True, slots=True)
class PredictionOutcomeResolver:
    """
    Connects stored predictions with later evidence and diagnostic history reads.

    Does not create Problems or write incident truth — only classifies prediction outcomes.
    """

    def resolve(
        self,
        record: PredictiveRiskHistoryRecord,
        future: PredictionFutureEvidenceSnapshot,
    ) -> PredictionOutcomeResolution:
        window_end = record.generated_at + timedelta(
            seconds=record.prediction_window.duration_seconds,
        )

        if future.problem_created_for_subject and _risk_aligns(record, future):
            return PredictionOutcomeResolution(
                outcome_status=PredictiveHistoryOutcomeStatus.CONFIRMED,
                rationale=(
                    "Diagnostic history shows a Problem for the predicted subject "
                    "after the risk signal."
                ),
            )

        if future.execution_failed and _evidence_overlap(record, future):
            return PredictionOutcomeResolution(
                outcome_status=PredictiveHistoryOutcomeStatus.CONFIRMED,
                rationale="Future execution failure evidence overlaps prediction refs.",
            )

        if future.observed_at > window_end:
            if not future.evidence_refs and not future.execution_failed:
                return PredictionOutcomeResolution(
                    outcome_status=PredictiveHistoryOutcomeStatus.EXPIRED,
                    rationale="Prediction window elapsed without corroborating evidence.",
                )
            if future.execution_failed or future.problem_created_for_subject:
                if not _risk_aligns(record, future):
                    return PredictionOutcomeResolution(
                        outcome_status=PredictiveHistoryOutcomeStatus.FALSE_POSITIVE,
                        rationale="Incident scope does not align with predicted risk subject.",
                    )
            return PredictionOutcomeResolution(
                outcome_status=PredictiveHistoryOutcomeStatus.FALSE_POSITIVE,
                rationale="Observed facts do not support the predicted risk within window.",
            )

        if not future.evidence_refs:
            return PredictionOutcomeResolution(
                outcome_status=PredictiveHistoryOutcomeStatus.INSUFFICIENT_EVIDENCE,
                rationale="No future evidence available yet within the prediction window.",
            )

        if _evidence_overlap(record, future):
            return PredictionOutcomeResolution(
                outcome_status=PredictiveHistoryOutcomeStatus.CONFIRMED,
                rationale="Corroborating evidence appeared within the prediction window.",
            )

        return PredictionOutcomeResolution(
            outcome_status=PredictiveHistoryOutcomeStatus.UNKNOWN,
            rationale="Evidence present but inconclusive for this prediction.",
        )


def _evidence_overlap(
    record: PredictiveRiskHistoryRecord,
    future: PredictionFutureEvidenceSnapshot,
) -> bool:
    prior = set(record.evidence_refs)
    return any(ref in prior for ref in future.evidence_refs)


def _risk_aligns(
    record: PredictiveRiskHistoryRecord,
    future: PredictionFutureEvidenceSnapshot,
) -> bool:
    subject = record.subject_identity.lower()
    risk = record.risk_type.lower()
    if subject in {kw.lower() for kw in future.matching_risk_keywords}:
        return True
    return any(
        subject in kw.lower() or risk in kw.lower() for kw in future.matching_risk_keywords
    )


__all__ = [
    "PredictionFutureEvidenceSnapshot",
    "PredictionOutcomeResolution",
    "PredictionOutcomeResolver",
]
