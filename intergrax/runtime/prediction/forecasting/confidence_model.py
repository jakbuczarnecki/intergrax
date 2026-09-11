# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Evidence-based confidence composition (PREDICTIVE R3)."""

from __future__ import annotations


def compose_forecast_confidence(
    *,
    evidence_strength: float,
    analyzer_reliability: float,
    data_completeness: float,
) -> float:
    """
    Confidence = evidence strength × historical analyzer reliability × data completeness.

    All inputs must already be bounded to [0.0, 1.0].
    """
    raw = evidence_strength * analyzer_reliability * data_completeness
    return min(1.0, max(0.0, raw))


__all__ = ["compose_forecast_confidence"]
