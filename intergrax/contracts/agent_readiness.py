# © Artur Czarnecki. All rights reserved.

"""Agent production readiness scoreboard contracts (architecture §40.15 · ACP-PROD-12)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentReadinessDimension(str, Enum):
    CONTRACT = "contract"
    RUNTIME = "runtime"
    POLICY = "policy"
    OBSERVABILITY = "observability"
    CHECKPOINTING = "checkpointing"
    IDEMPOTENCY = "idempotency"
    SECURITY = "security"
    EVALUATION = "evaluation"
    LIFECYCLE = "lifecycle"
    CAPABILITY_ROUTING = "capability_routing"


class AgentReadinessStatus(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


DEFAULT_DIMENSION_WEIGHTS: dict[AgentReadinessDimension, float] = {
    AgentReadinessDimension.CONTRACT: 0.10,
    AgentReadinessDimension.RUNTIME: 0.15,
    AgentReadinessDimension.POLICY: 0.10,
    AgentReadinessDimension.OBSERVABILITY: 0.10,
    AgentReadinessDimension.CHECKPOINTING: 0.10,
    AgentReadinessDimension.IDEMPOTENCY: 0.10,
    AgentReadinessDimension.SECURITY: 0.10,
    AgentReadinessDimension.EVALUATION: 0.10,
    AgentReadinessDimension.LIFECYCLE: 0.05,
    AgentReadinessDimension.CAPABILITY_ROUTING: 0.10,
}


class AgentReadinessDimensionScore(BaseModel):
    """One scored readiness dimension for an agent."""

    model_config = ConfigDict(extra="forbid")

    dimension: AgentReadinessDimension
    pct: float = Field(ge=0.0, le=100.0)
    status: AgentReadinessStatus
    weight: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


class AgentProductionReadinessReport(BaseModel):
    """Typed production readiness scoreboard row (ACP-PROD-12)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_production_readiness.v1"] = "agent_production_readiness.v1"
    agent_id: str
    contract_id: str
    generated_at: datetime
    overall_pct: float = Field(ge=0.0, le=100.0)
    production_eligible_recommendation: bool
    dimensions: list[AgentReadinessDimensionScore]

    def dimension_score(self, dimension: AgentReadinessDimension) -> AgentReadinessDimensionScore | None:
        for item in self.dimensions:
            if item.dimension == dimension:
                return item
        return None


class AgentProductionReadinessRosterReport(BaseModel):
    """Roster-wide scoreboard artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_production_readiness_roster.v1"] = (
        "agent_production_readiness_roster.v1"
    )
    generated_at: datetime
    agent_count: int
    roster_mean_overall_pct: float
    runtime_dimension_mean_pct: float
    fleet_migration_complete: bool
    agents: list[AgentProductionReadinessReport]
