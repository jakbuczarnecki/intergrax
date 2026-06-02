# © Artur Czarnecki. All rights reserved.

"""Retrieval poisoning defense contracts (Phase V-SEC.3)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RetrievalTrustLevel(str, Enum):
    TRUSTED = "trusted"
    REVIEW = "review"
    QUARANTINED = "quarantined"


class RetrievalDocumentSignal(BaseModel):
    document_id: str
    trust_score: float
    source_ref: str


class RetrievalPoisoningDecision(BaseModel):
    document_id: str
    trust_level: RetrievalTrustLevel
    quarantined: bool
    reasons: list[str] = Field(default_factory=list)


class RetrievalPoisoningReport(BaseModel):
    schema_version: str = "1.0.0"
    decisions: list[RetrievalPoisoningDecision] = Field(default_factory=list)


def evaluate_retrieval_poisoning(
    *,
    signals: list[RetrievalDocumentSignal],
    quarantine_threshold: float = 0.40,
    review_threshold: float = 0.70,
) -> RetrievalPoisoningReport:
    decisions: list[RetrievalPoisoningDecision] = []
    for signal in signals:
        if signal.trust_score < quarantine_threshold:
            decisions.append(
                RetrievalPoisoningDecision(
                    document_id=signal.document_id,
                    trust_level=RetrievalTrustLevel.QUARANTINED,
                    quarantined=True,
                    reasons=["Trust score below quarantine threshold"],
                )
            )
            continue
        if signal.trust_score < review_threshold:
            decisions.append(
                RetrievalPoisoningDecision(
                    document_id=signal.document_id,
                    trust_level=RetrievalTrustLevel.REVIEW,
                    quarantined=False,
                    reasons=["Trust score requires manual review"],
                )
            )
            continue
        decisions.append(
            RetrievalPoisoningDecision(
                document_id=signal.document_id,
                trust_level=RetrievalTrustLevel.TRUSTED,
                quarantined=False,
                reasons=[],
            )
        )
    return RetrievalPoisoningReport(decisions=decisions)
