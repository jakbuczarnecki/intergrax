# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Prediction audit chain contracts (PREDICTIVE R4 governance)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.predictive.quality import PredictiveQualityAssessment


@dataclass(frozen=True, slots=True)
class PredictionAuditRecord:
    """Audit envelope for one prediction run — not diagnostic authority."""

    prediction_run_id: str
    tenant_id: str
    context_snapshot_id: str
    input_snapshot_id: str
    generated_at: datetime
    analyzer_ids: tuple[str, ...]
    analyzer_versions: tuple[str, ...]
    provider_versions: tuple[str, ...]
    quality_assessment: PredictiveQualityAssessment
    signal_ids: tuple[str, ...]
    analyzer_outcomes: tuple[str, ...]
    degraded: bool

    @property
    def prediction_id(self) -> str:
        """Backward-compatible alias."""
        return self.prediction_run_id

    @property
    def model_versions(self) -> tuple[str, ...]:
        """Backward-compatible alias."""
        return self.analyzer_versions


__all__ = ["PredictionAuditRecord"]
