# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Persisted outcome evaluations for reconstruction (PREDICTIVE R5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from intergrax.contracts.predictive.outcome.evaluation import PredictionOutcomeEvaluation


class PredictionOutcomePersistence(Protocol):
    def append(self, evaluation: PredictionOutcomeEvaluation) -> PredictionOutcomeEvaluation: ...

    def list_for_signal(
        self,
        *,
        tenant_id: str,
        prediction_signal_id: str,
    ) -> tuple[PredictionOutcomeEvaluation, ...]: ...

    def list_for_tenant(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> tuple[PredictionOutcomeEvaluation, ...]: ...


@dataclass
class InMemoryPredictionOutcomePersistence:
    """Tenant-scoped in-process store — not incident authority."""

    _rows: list[PredictionOutcomeEvaluation] = field(default_factory=list)

    def append(self, evaluation: PredictionOutcomeEvaluation) -> PredictionOutcomeEvaluation:
        if evaluation.tenant_id.strip() == "":
            raise ValueError("tenant_id required")
        self._rows.append(evaluation)
        return evaluation

    def list_for_signal(
        self,
        *,
        tenant_id: str,
        prediction_signal_id: str,
    ) -> tuple[PredictionOutcomeEvaluation, ...]:
        return tuple(
            row
            for row in self._rows
            if row.tenant_id == tenant_id and row.prediction_signal_id == prediction_signal_id
        )

    def list_for_tenant(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> tuple[PredictionOutcomeEvaluation, ...]:
        matched = [row for row in self._rows if row.tenant_id == tenant_id]
        return tuple(matched[-limit:])


__all__ = [
    "InMemoryPredictionOutcomePersistence",
    "PredictionOutcomePersistence",
]
