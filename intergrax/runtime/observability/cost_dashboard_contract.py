# © Artur Czarnecki. All rights reserved.

"""Cost dashboard metrics contract (IDEAL-21.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CostDashboardMetrics(BaseModel):
    tenant_id: str
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    run_count: int = 0
    labels: dict[str, str] = Field(default_factory=dict)
