# © Artur Czarnecki. All rights reserved.

"""Verification loop contracts for L4-V (Phase W-ADAPT-5)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.runtime.adaptive.contracts import ProfileArtifactType
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
from intergrax.runtime.architecture.cost_budget import BudgetEnvelope
from intergrax.runtime.architecture.evaluation_registry_trends import (
    EvaluationRegistryTrendReport,
)


class VerificationCheckId(str, Enum):
    """Individual verification gate identifiers (AHIA §9.6)."""

    UTILITY_TREND = "utility_trend"
    EVAL_REGISTRY = "eval_registry"
    REGRESSION_RATE = "regression_rate"
    COST_BUDGET = "cost_budget"
    SECURITY_ADVERSARIAL = "security_adversarial"


class VerificationCheckResult(BaseModel):
    """Outcome of a single verification check."""

    model_config = ConfigDict(extra="forbid")

    check_id: VerificationCheckId
    passed: bool
    detail: str = ""
    metric_value: float | None = None
    baseline_value: float | None = None


class VerificationTarget(BaseModel):
    """Profile version under post-apply verification."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    artifact_type: ProfileArtifactType
    candidate_version_id: str
    loop_id: str | None = None
    loop_kind: AdaptiveLoopKind | None = None


class VerificationContext(BaseModel):
    """Typed inputs for a verification cycle."""

    model_config = ConfigDict(extra="forbid")

    evaluation_trend: EvaluationRegistryTrendReport | None = None
    budget_envelopes: list[BudgetEnvelope] = Field(default_factory=list)
    min_improvement_delta: float = Field(default=0.0, ge=0.0)
    min_utility_improvement_ratio: float = Field(default=0.10, ge=0.0)
    max_regression_rate_delta: float = Field(default=0.10, ge=0.0)
    max_cost_normalized: float = Field(default=1.0, ge=0.0)
    min_run_count: int = Field(default=3, ge=1)
    window_days: int = Field(default=7, ge=1)
    auto_rollback_enabled: bool = True
    auto_rollback_service_principal: RequestIdentity | None = None
    auto_rollback_mutation_id: str | None = None


class VerificationResult(BaseModel):
    """Verification outcome for one profile target."""

    model_config = ConfigDict(extra="forbid")

    target: VerificationTarget
    passed: bool
    checks: list[VerificationCheckResult] = Field(default_factory=list)
    rolled_back: bool = False
    blocked_loop_kind: AdaptiveLoopKind | None = None
    failure_reasons: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Batch verification report exported to ops artifacts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    results: list[VerificationResult] = Field(default_factory=list)
    passed: bool = False
    rollback_count: int = 0
    blocked_loop_kinds: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
