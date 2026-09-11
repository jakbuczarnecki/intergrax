# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Observed statistical features for forecasting — no Problem or root-cause fields (R3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SubjectPredictiveFeatures:
    """Bounded numeric observations for one subject identity."""

    subject_identity: str
    latency_growth_rate: float | None = None
    failure_rate: float | None = None
    failure_frequency_delta: float | None = None
    retry_per_execution: float | None = None
    retry_growth_rate: float | None = None
    execution_variance: float | None = None
    resource_utilization_latest: float | None = None
    resource_utilization_slope: float | None = None
    data_completeness: float = 0.0

    def __post_init__(self) -> None:
        if not self.subject_identity.strip():
            raise ValueError("subject_identity must be non-empty")
        if not (0.0 <= self.data_completeness <= 1.0):
            raise ValueError("data_completeness must be in [0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class PredictiveFeatureSet:
    """Immutable feature bundle — observations only."""

    tenant_id: str
    input_snapshot_id: str
    as_of: datetime
    subjects: tuple[SubjectPredictiveFeatures, ...]
    global_data_completeness: float

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.input_snapshot_id.strip():
            raise ValueError("input_snapshot_id must be non-empty")
        if not (0.0 <= self.global_data_completeness <= 1.0):
            raise ValueError("global_data_completeness must be in [0.0, 1.0]")


__all__ = ["PredictiveFeatureSet", "SubjectPredictiveFeatures"]
