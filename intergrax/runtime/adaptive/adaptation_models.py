# © Artur Czarnecki. All rights reserved.

"""Adaptation engine models for L4-R recommend wave (Phase W-ADAPT-2)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.contracts import (
    ADAPTIVE_PACKAGE_SCHEMA_VERSION,
    HarnessOutcomeSignal,
    ProfileVersionDraft,
)
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveLoopGateResult,
    AdaptiveLoopProposal,
)
from intergrax.runtime.architecture.capability_graph import CapabilityGraph
from intergrax.runtime.architecture.capability_graph_compatibility import (
    CapabilityCompatibilityReport,
)
from intergrax.runtime.architecture.cost_forecast import CostAnomalyRecord
from intergrax.runtime.architecture.evaluation_registry_trends import (
    EvaluationRegistryTrendReport,
)


class BanditArmState(BaseModel):
    """Thompson sampling state for a contextual bandit arm."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    arm_id: str
    alpha: float = Field(default=1.0, ge=0.0)
    beta: float = Field(default=1.0, ge=0.0)
    observation_count: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AdaptationProposalCandidate(BaseModel):
    """Raw proposal candidate emitted by a sub-engine before governance wrapping."""

    model_config = ConfigDict(extra="forbid")

    loop_id: str
    source_engine: str
    proposal: AdaptiveLoopProposal
    profile_draft: ProfileVersionDraft | None = None
    rank_score: float = 0.0
    cooldown_seconds: int = Field(default=300, ge=0)


class AdaptationProposalPackage(BaseModel):
    """Governed proposal package ready for persistence and ops review."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(default_factory=lambda: f"prop_{uuid4().hex}")
    candidate: AdaptationProposalCandidate
    envelope_gate: AdaptiveLoopGateResult
    capability_gate_passed: bool = True
    capability_report: CapabilityCompatibilityReport | None = None
    golden_scenario_gate_passed: bool = True
    passed_all_gates: bool = False
    gate_reasons: list[str] = Field(default_factory=list)


class AdaptationEngineContext(BaseModel):
    """Typed inputs for a single adaptation engine cycle."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    task_class: str
    signals: list[HarnessOutcomeSignal] = Field(default_factory=list)
    evaluation_trend: EvaluationRegistryTrendReport | None = None
    cost_anomalies: list[CostAnomalyRecord] = Field(default_factory=list)
    capability_graph_previous: CapabilityGraph | None = None
    capability_graph_candidate: CapabilityGraph | None = None
    golden_scenario_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    golden_scenario_min_pass_rate: float = Field(default=0.70, ge=0.0, le=1.0)
    default_human_approver_id: str = "owner:harness-ops"


class AdaptationEngineRunResult(BaseModel):
    """Outcome of a recommend-only adaptation engine cycle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = ADAPTIVE_PACKAGE_SCHEMA_VERSION
    tenant_id: str
    task_class: str
    packages: list[AdaptationProposalPackage] = Field(default_factory=list)
    skipped_cooldown_loop_ids: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
