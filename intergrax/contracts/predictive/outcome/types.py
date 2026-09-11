# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Extensible outcome vocabulary for prediction feedback (PREDICTIVE R5)."""

from __future__ import annotations

from enum import StrEnum


class PredictionOutcomeType(StrEnum):
    """Domain-extensible outcome facts — must be backed by evidence refs."""

    INCIDENT_OCCURRED = "INCIDENT_OCCURRED"
    NO_INCIDENT = "NO_INCIDENT"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    TRUE_POSITIVE = "TRUE_POSITIVE"
    UNKNOWN = "UNKNOWN"


class PredictionOutcomeEvaluationStatus(StrEnum):
    """Whether an evaluation pass completed and how."""

    EVALUATED = "EVALUATED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    RESOLVER_FAILED = "RESOLVER_FAILED"
    RESOLVER_TIMEOUT = "RESOLVER_TIMEOUT"
    UNKNOWN = "UNKNOWN"


__all__ = [
    "PredictionOutcomeEvaluationStatus",
    "PredictionOutcomeType",
]
