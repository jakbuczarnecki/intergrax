# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Feature extraction SPI — separates observation from analysis (PREDICTIVE R3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.contracts.predictive_feature_set import PredictiveFeatureSet


@runtime_checkable
class PredictiveFeatureExtractor(Protocol):
    """Derive statistical features from readonly predictive context."""

    def extract(self, context: PredictiveContext) -> PredictiveFeatureSet:
        """Pure extraction — no persistence or Problem authority."""


__all__ = ["PredictiveFeatureExtractor"]
