# © Artur Czarnecki. All rights reserved.

"""Quality / governance / cost health dashboard contracts (AUDIT-IDEAL-21.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QualityHealthSnapshot(BaseModel):
    """Critic and evaluation quality signals for dashboard consumers."""

    schema_version: str = "1.0.0"
    critic_pass_rate: float = Field(ge=0.0, le=1.0)
    shadow_eval_coverage_ratio: float = Field(ge=0.0, le=1.0)
    open_human_review_count: int = Field(ge=0)


class GovernanceHealthSnapshot(BaseModel):
    """Policy and prompt governance posture."""

    schema_version: str = "1.0.0"
    policy_denial_rate: float = Field(ge=0.0, le=1.0)
    prompt_approval_pending_count: int = Field(ge=0)
    tenant_isolation_verified: bool


class CostHealthSnapshot(BaseModel):
    """Budget utilization and optimization posture."""

    schema_version: str = "1.0.0"
    budget_utilization_ratio: float = Field(ge=0.0, le=2.0)
    forecast_anomaly_count: int = Field(ge=0)
    optimization_recommendation_count: int = Field(ge=0)


class HarnessHealthDashboardContract(BaseModel):
    """Unified health dashboard payload for Tier-3 ops surfaces."""

    schema_version: str = "1.0.0"
    host_id: str
    quality: QualityHealthSnapshot
    governance: GovernanceHealthSnapshot
    cost: CostHealthSnapshot


def build_harness_health_dashboard_contract(
    *,
    host_id: str,
    critic_pass_rate: float = 1.0,
    shadow_eval_coverage_ratio: float = 0.0,
    open_human_review_count: int = 0,
    policy_denial_rate: float = 0.0,
    prompt_approval_pending_count: int = 0,
    tenant_isolation_verified: bool = True,
    budget_utilization_ratio: float = 0.0,
    forecast_anomaly_count: int = 0,
    optimization_recommendation_count: int = 0,
) -> HarnessHealthDashboardContract:
    """Build a typed health dashboard contract from host-level signals."""
    return HarnessHealthDashboardContract(
        host_id=host_id,
        quality=QualityHealthSnapshot(
            critic_pass_rate=critic_pass_rate,
            shadow_eval_coverage_ratio=shadow_eval_coverage_ratio,
            open_human_review_count=open_human_review_count,
        ),
        governance=GovernanceHealthSnapshot(
            policy_denial_rate=policy_denial_rate,
            prompt_approval_pending_count=prompt_approval_pending_count,
            tenant_isolation_verified=tenant_isolation_verified,
        ),
        cost=CostHealthSnapshot(
            budget_utilization_ratio=budget_utilization_ratio,
            forecast_anomaly_count=forecast_anomaly_count,
            optimization_recommendation_count=optimization_recommendation_count,
        ),
    )
