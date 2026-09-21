# © Artur Czarnecki. All rights reserved.

"""Configurable adaptive utility weights (AHIA §10.2) — public declarative contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UtilityWeights(BaseModel):
    """Configurable utility weights (AHIA §10.2)."""

    model_config = ConfigDict(extra="forbid")

    w_quality: float = Field(default=0.50, ge=0.0, le=1.0)
    w_cost: float = Field(default=0.25, ge=0.0, le=1.0)
    w_latency: float = Field(default=0.10, ge=0.0, le=1.0)
    w_hitl: float = Field(default=0.10, ge=0.0, le=1.0)
    w_regression: float = Field(default=0.05, ge=0.0, le=1.0)
    w_business: float = Field(default=0.00, ge=0.0, le=1.0)
    latency_slo_ms: int = Field(default=30_000, ge=1)
    max_hitl_interventions: int = Field(default=3, ge=1)
