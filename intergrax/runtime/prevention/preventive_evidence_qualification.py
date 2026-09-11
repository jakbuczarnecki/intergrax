# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Evidence quality labeling for preventive recommendations (PREVENTIVE R6-Q)."""

from __future__ import annotations

from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport


def evidence_quality_label(context_quality: PredictiveContextQualityReport) -> str:
    score = min(context_quality.coverage, context_quality.reliability)
    if score >= 0.85:
        return "HIGH"
    if score >= 0.6:
        return "MEDIUM"
    return "LOW"


__all__ = ["evidence_quality_label"]
