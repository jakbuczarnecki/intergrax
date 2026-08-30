# © Artur Czarnecki. All rights reserved.

"""Unified product observability dashboard payload (AUDIT-IDEAL-21.3 / GOV-PROD.1)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.observability.health_dashboard_contracts import HarnessHealthDashboardContract


class ArchitectureHealthPane(BaseModel):
    """Modularity and debt signals for ops dashboard consumers."""

    schema_version: str = "1.0.0"
    modularity_score: float = Field(ge=0.0, le=1.0)
    debt_index: float = Field(ge=0.0, le=1.0)
    capability_count: int = Field(ge=0)


class DiagnosticOperationsPane(BaseModel):
    """Central diagnostic capability summary for the product host."""

    schema_version: str = "1.0.0"
    ready: bool
    problem_count: int | None = Field(default=None, ge=0)
    open_problem_count: int | None = Field(default=None, ge=0)
    problem_count_is_exact: bool = False
    open_problem_count_is_exact: bool = False


class GovernanceDashboardPane(BaseModel):
    """Governance health slice (AUDIT-IDEAL-5.3)."""

    schema_version: str = "1.0.0"
    compliance_profile_enabled: bool
    strategy_review_enabled: bool
    tenant_isolation_verified: bool
    policy_denial_rate: float = Field(ge=0.0, le=1.0)
    prompt_approval_pending_count: int = Field(ge=0)


class ProductObservabilityDashboard(BaseModel):
    """GOV-PROD.1 unified dashboard contract for Tier-3 product hosts."""

    schema_version: str = "1.0.0"
    host_id: str
    governance: GovernanceDashboardPane
    health: HarnessHealthDashboardContract
    architecture: ArchitectureHealthPane
    diagnostics: DiagnosticOperationsPane


def build_product_observability_dashboard(
    *,
    host_id: str,
    governance: GovernanceDashboardPane,
    health: HarnessHealthDashboardContract,
    architecture: ArchitectureHealthPane,
    diagnostics: DiagnosticOperationsPane,
) -> ProductObservabilityDashboard:
    """Assemble the unified observability dashboard payload."""
    return ProductObservabilityDashboard(
        host_id=host_id,
        governance=governance,
        health=health,
        architecture=architecture,
        diagnostics=diagnostics,
    )
