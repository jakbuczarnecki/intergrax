# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Prediction history persistence port — never Problem persistence (PREDICTIVE R2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
)


class PredictiveHistoryPersistenceConflictError(Exception):
    """Raised when a history write conflicts with an existing record."""


class PredictiveHistoryPersistence(ABC):
    """
    Plugin-neutral port for prediction lifecycle memory.

    Implementations must not delegate to diagnostic Problem stores or incident stores.
    """

    @abstractmethod
    def append(self, record: PredictiveRiskHistoryRecord) -> PredictiveRiskHistoryRecord:
        """Persist a new prediction history row (typically CREATED)."""

    @abstractmethod
    def get(
        self,
        *,
        tenant_id: str,
        risk_signal_id: str,
    ) -> PredictiveRiskHistoryRecord | None:
        """Load one history row by tenant and signal id."""

    @abstractmethod
    def update_outcome(
        self,
        *,
        tenant_id: str,
        risk_signal_id: str,
        outcome_status: PredictiveHistoryOutcomeStatus,
    ) -> PredictiveRiskHistoryRecord:
        """Transition outcome with lifecycle validation."""

    @abstractmethod
    def list_for_tenant(
        self,
        *,
        tenant_id: str,
        subject_identity: str | None = None,
        limit: int = 100,
    ) -> tuple[PredictiveRiskHistoryRecord, ...]:
        """Bounded tenant-scoped listing for read models and metrics."""


__all__ = [
    "PredictiveHistoryPersistence",
    "PredictiveHistoryPersistenceConflictError",
]
