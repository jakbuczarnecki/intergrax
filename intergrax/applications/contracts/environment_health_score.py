# © Artur Czarnecki. All rights reserved.

"""Platform-scoped Tier-3 environment health score contracts (APP-OPS-3 · §50.3)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class HealthDimension(StrEnum):
    """Ops health dimensions for application environments (§50.3.1)."""

    DEPRECATED_CAPABILITIES = "deprecated_capabilities"
    STALE_AGENTS = "stale_agents"
    FAILED_MIGRATIONS = "failed_migrations"
    POLICY_COVERAGE = "policy_coverage"
    TEST_COVERAGE = "test_coverage"
    OWNERSHIP_COMPLETE = "ownership_complete"
    CAPABILITY_GRAPH_VALID = "capability_graph_valid"
    BUDGET_GOVERNANCE_CONFIGURED = "budget_governance_configured"
    RECOVERY_CONTRACT_DOCUMENTED = "recovery_contract_documented"


PRODUCTION_READY_THRESHOLD = 0.9


class HealthDimensionScore(BaseModel):
    """Score for one health dimension."""

    model_config = ConfigDict(extra="forbid")

    dimension: HealthDimension
    score: float = Field(ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    stale_after: datetime | None = None


class EnvironmentHealthScore(BaseModel):
    """Continuous ops health score for one deployed environment."""

    model_config = ConfigDict(extra="forbid")

    app_id: str
    environment_id: str
    snapshot_id: str | None = None
    scored_at: datetime
    overall: float = Field(ge=0.0, le=1.0)
    dimensions: list[HealthDimensionScore] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ApplicationHealthScore(BaseModel):
    """Rollup across registered environments for one application (§50.3.2)."""

    model_config = ConfigDict(extra="forbid")

    app_id: str
    environments: list[EnvironmentHealthScore] = Field(default_factory=list)
    worst_environment: str | None = None
    production_ready: bool = False
