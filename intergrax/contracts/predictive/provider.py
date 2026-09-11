# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive context provider SPI (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intergrax.contracts.predictive.provenance import PredictiveContextProvenance
from intergrax.contracts.predictive.sections import (
    PredictiveContextDiagnostic,
    PredictiveContextHistory,
    PredictiveContextPerformance,
)
from intergrax.contracts.predictive.scope import PredictiveScope


class PredictiveContextProviderStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True, slots=True)
class PredictiveContextProviderFragment:
    """One provider contribution — fragments only, no cross-provider logic."""

    provider_id: str
    status: PredictiveContextProviderStatus
    history: PredictiveContextHistory | None = None
    performance: PredictiveContextPerformance | None = None
    diagnostic: PredictiveContextDiagnostic | None = None
    current_state: tuple[str, ...] = ()
    decision_history: tuple[str, ...] = ()
    lineage_patterns: tuple[str, ...] = ()
    provenance: PredictiveContextProvenance | None = None


class PredictiveContextProvider(Protocol):
    """Build one bounded slice of predictive context for a scope."""

    provider_id: str
    provider_version: str

    def build(self, scope: PredictiveScope) -> PredictiveContextProviderFragment:
        ...


__all__ = [
    "PredictiveContextProvider",
    "PredictiveContextProviderFragment",
    "PredictiveContextProviderStatus",
]
