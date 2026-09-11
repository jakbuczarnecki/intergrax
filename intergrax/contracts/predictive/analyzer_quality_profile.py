# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Analyzer quality intelligence profile (PREDICTIVE R4 governance, R5 learning)."""

from __future__ import annotations

from dataclasses import dataclass


def _derive_rates(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
    cold_precision: float,
) -> tuple[float, float, float, float, float]:
    decided_pos = true_positive + false_positive
    precision = (true_positive / decided_pos) if decided_pos else cold_precision
    recall_denom = true_positive + false_negative
    recall = (true_positive / recall_denom) if recall_denom else precision
    fp_denom = false_positive + true_negative
    false_positive_rate = (false_positive / fp_denom) if fp_denom else 0.0
    false_negative_rate = (false_negative / recall_denom) if recall_denom else 0.0
    confidence_calibration = precision
    return precision, recall, false_positive_rate, false_negative_rate, confidence_calibration


@dataclass(frozen=True, slots=True)
class PredictiveAnalyzerQualityProfile:
    """Historical effectiveness for one analyzer within a tenant scope."""

    analyzer_id: str
    tenant_id: str
    predictions: int
    true_positive: int
    false_positive: int
    false_negative: int = 0
    true_negative: int = 0
    precision: float = 0.75
    recall: float = 0.75
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    confidence_calibration: float = 0.75

    def __post_init__(self) -> None:
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if self.predictions < 0:
            raise ValueError("predictions must be non-negative")
        for name, value in (
            ("true_positive", self.true_positive),
            ("false_positive", self.false_positive),
            ("false_negative", self.false_negative),
            ("true_negative", self.true_negative),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("precision", self.precision),
            ("recall", self.recall),
            ("false_positive_rate", self.false_positive_rate),
            ("false_negative_rate", self.false_negative_rate),
            ("confidence_calibration", self.confidence_calibration),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0]")


def default_analyzer_quality_profile(
    *,
    analyzer_id: str,
    tenant_id: str,
) -> PredictiveAnalyzerQualityProfile:
    """Cold-start profile — conservative until feedback arrives."""
    return PredictiveAnalyzerQualityProfile(
        analyzer_id=analyzer_id,
        tenant_id=tenant_id,
        predictions=0,
        true_positive=0,
        false_positive=0,
        false_negative=0,
        true_negative=0,
        precision=0.75,
        recall=0.75,
        false_positive_rate=0.0,
        false_negative_rate=0.0,
        confidence_calibration=0.75,
    )


def profile_after_outcome_counts(
    profile: PredictiveAnalyzerQualityProfile,
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
    predictions: int,
) -> PredictiveAnalyzerQualityProfile:
    precision, recall, fpr, fnr, calibration = _derive_rates(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        cold_precision=profile.precision,
    )
    return PredictiveAnalyzerQualityProfile(
        analyzer_id=profile.analyzer_id,
        tenant_id=profile.tenant_id,
        predictions=predictions,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=precision,
        recall=recall,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        confidence_calibration=calibration,
    )


__all__ = [
    "PredictiveAnalyzerQualityProfile",
    "default_analyzer_quality_profile",
    "profile_after_outcome_counts",
]
