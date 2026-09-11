# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive context intelligence layer entrypoint (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive import PredictiveContext, PredictiveScope
from intergrax.runtime.prediction.context.predictive_context_aggregator import (
    PredictiveContextAggregator,
)


@dataclass(frozen=True, slots=True)
class PredictiveContextBuilder:
    """Aggregate diagnostic read-model providers into one bounded context."""

    aggregator: PredictiveContextAggregator

    def build(self, scope: PredictiveScope) -> PredictiveContext:
        return self.aggregator.aggregate(scope)


__all__ = ["PredictiveContextBuilder"]
