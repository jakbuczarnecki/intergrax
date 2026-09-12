# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Audit metadata carried with autonomy evaluation (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AutonomyAuditBundle:
    """Immutable audit snapshot attached to autonomy outcomes."""

    tenant_id: str
    diagnostic_investigation_id: str
    problem_id: str
    recommendation_correlation_id: str
    recorded_at: datetime
    principal_id: str | None = None
    trace_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.diagnostic_investigation_id.strip():
            raise ValueError("diagnostic_investigation_id required")
        if not self.problem_id.strip():
            raise ValueError("problem_id required")
        if not self.recommendation_correlation_id.strip():
            raise ValueError("recommendation_correlation_id required")


__all__ = ["AutonomyAuditBundle"]
