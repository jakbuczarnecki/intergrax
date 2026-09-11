# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive analysis scope (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictiveScope:
    """Tenant-safe bounded scope — not diagnostic authority."""

    tenant_id: str
    task_id: str | None = None
    run_id: str | None = None
    execution_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")


__all__ = ["PredictiveScope"]
