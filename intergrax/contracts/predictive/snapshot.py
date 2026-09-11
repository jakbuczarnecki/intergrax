# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Immutable predictive context snapshots (PREDICTIVE R4 governance)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.predictive.context import PredictiveContext


@dataclass(frozen=True, slots=True)
class PredictiveContextSnapshot:
    """Frozen context at prediction time — survives later read-model drift."""

    snapshot_id: str
    captured_at: datetime
    context: PredictiveContext

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty")


__all__ = ["PredictiveContextSnapshot"]
