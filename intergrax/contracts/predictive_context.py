# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Readonly predictive analysis context contract (PREDICTIVE R1/R4)."""

from intergrax.contracts.predictive import (
    ExecutionPatternSnapshot,
    HistoricalProblemRef,
    PerformanceMetricPoint,
    PredictiveContext,
    PredictiveContextCompleteness,
    PredictiveContextMetadata,
    PredictiveScope,
    PREDICTIVE_CONTEXT_VERSION,
    predictive_context_from_legacy_fields,
)

__all__ = [
    "ExecutionPatternSnapshot",
    "HistoricalProblemRef",
    "PREDICTIVE_CONTEXT_VERSION",
    "PerformanceMetricPoint",
    "PredictiveContext",
    "PredictiveContextCompleteness",
    "PredictiveContextMetadata",
    "PredictiveScope",
    "predictive_context_from_legacy_fields",
]
