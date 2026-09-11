# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Registry governance metadata for predictive plugins (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.forecast_analyzer_descriptor import ForecastResourceBudget
from intergrax.contracts.predictive.analyzer_quality_profile import (
    PredictiveAnalyzerQualityProfile,
)


@dataclass(frozen=True, slots=True)
class PredictivePluginRegistration:
    """Enterprise registry row — deterministic ordering and audit identity."""

    plugin_id: str
    namespace: str
    version: str
    owner: str
    capabilities: tuple[str, ...]
    quality_profile: PredictiveAnalyzerQualityProfile
    resource_budget: ForecastResourceBudget
    priority: int

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("plugin_id must be non-empty")
        if not self.namespace.strip():
            raise ValueError("namespace must be non-empty")
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.owner.strip():
            raise ValueError("owner must be non-empty")
        if not self.capabilities:
            raise ValueError("capabilities must be non-empty")


__all__ = ["PredictivePluginRegistration"]
