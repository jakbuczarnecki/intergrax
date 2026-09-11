# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Statistical forecast analyzer SPI — deterministic bounded plugins (PREDICTIVE R3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.forecast_analyzer_descriptor import ForecastAnalyzerDescriptor
from intergrax.contracts.predictive_feature_set import PredictiveFeatureSet
from intergrax.contracts.predictive_historical_intelligence import HistoricalRiskIntelligence
from intergrax.contracts.predictive_risk import PredictiveRiskSignal


@runtime_checkable
class StatisticalForecastAnalyzer(Protocol):
    """Evidence-driven statistical forecast — no Problem writes."""

    @property
    def descriptor(self) -> ForecastAnalyzerDescriptor:
        """Governance identity for registry ordering and audit."""

    def analyze(
        self,
        features: PredictiveFeatureSet,
        *,
        historical_intelligence: HistoricalRiskIntelligence,
    ) -> tuple[PredictiveRiskSignal, ...]:
        """Pure analysis over readonly features."""


__all__ = ["StatisticalForecastAnalyzer"]
