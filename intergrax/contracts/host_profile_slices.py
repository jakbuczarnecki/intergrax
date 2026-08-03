# © Artur Czarnecki. All rights reserved.

"""Minimal host profile slices referenced by runtime and agent merge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.agent_budget import BudgetReactionProfile
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.intent_route import IntentRoute
from intergrax.contracts.resilience_policy import ResiliencePolicy, default_resilience_policy
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.adaptive.contracts import UtilityWeights
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy

ContextEnginePreset = Literal[
    "default",
    "minimal",
    "long_session",
    "research_heavy",
    "tool_heavy",
]

AdaptiveMode = Literal["off", "observe", "recommend", "apply"]


class PolicyRulesProfile(BaseModel):
    """Declarative policy rules file reference."""

    model_config = ConfigDict(extra="forbid")

    rules_path: Path | None = None
    inline_rules: list[dict[str, Any]] = Field(default_factory=list)


class GuardrailProfile(BaseModel):
    """Vendor LLM guardrail scanning toggles."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    scan_input: bool = True
    scan_output: bool = True
    scan_tool_calls: bool = False
    secondary_slug: str | None = None
    colang_config_path: str | None = None
    bedrock_guardrail_policy_id: str | None = None
    inference_slug: str | None = None


class ContextDecisionProfile(BaseModel):
    """Unified memory vs context vs RAG assembly policy."""

    model_config = ConfigDict(extra="forbid")

    include_session_history: bool = True
    prefer_longterm_memory: bool = True
    prefer_rag_when_enabled: bool = True
    max_memory_entries_in_context: int = Field(default=8, ge=1, le=64)


class ContextProfile(BaseModel):
    """Context assembly defaults for Nexus."""

    model_config = ConfigDict(extra="ignore")

    assembly_options: TaskContextAssemblyOptions = Field(
        default_factory=TaskContextAssemblyOptions,
    )
    budget_policy: ContextBudgetPolicy | None = None
    decision: ContextDecisionProfile = Field(default_factory=ContextDecisionProfile)
    engine_preset: ContextEnginePreset = "default"
    engine_ref: str | None = None
    context_plugin_ids: list[str] = Field(default_factory=list)
    enable_rag: bool = True
    enable_websearch: bool = True
    drift_monitoring_enabled: bool = False
    drift_alert_threshold: float = Field(default=0.35, ge=0.0, le=2.0)
    optimization_policy: ContextOptimizationPolicy | None = None
    semantic_compression_enabled: bool = False
    default_history_compression: Literal["truncate_oldest", "summarize_oldest", "hybrid"] = (
        "truncate_oldest"
    )

    @field_validator("context_plugin_ids")
    @classmethod
    def _normalize_plugin_ids(cls, value: list[str]) -> list[str]:
        return [item.strip().lower() for item in value if item.strip()]

    @field_validator("engine_ref")
    @classmethod
    def _strip_engine_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class MemoryProfile(BaseModel):
    """Task/org memory flags for runtime merge."""

    model_config = ConfigDict(extra="ignore")

    enable_user_memory: bool = False
    enable_org_memory: bool = False
    enable_long_term_memory: bool = False
    enable_task_memory: bool = False


class CostProfile(BaseModel):
    """Run budget slice for agent session merge."""

    model_config = ConfigDict(extra="ignore")

    budget_reaction: BudgetReactionProfile | None = None
    max_total_tokens: int | None = Field(default=None, ge=1)


class ReliabilityProfile(BaseModel):
    """Idempotency, circuit breaker, checkpoint posture."""

    model_config = ConfigDict(extra="ignore")

    idempotency_enabled: bool = True
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    checkpoint_interval_steps: int = Field(default=1, ge=1)
    long_running_scheduler_enabled: bool = False
    resilience_policy: ResiliencePolicy = Field(default_factory=default_resilience_policy)
    default_autonomy_level: AutonomyLevel = AutonomyLevel.ASK
    tenant_autonomy_ceiling: AutonomyLevel | None = None


class ExecutionBoundaryExportProfile(BaseModel):
    """Execution boundary export settings for attestation runtime."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    capture_mode: str = Field(default="side_effects_only")
    allowlist: list[str] = Field(default_factory=list)
    include_canonical_io: bool = True
    step_level_enabled: bool = False
    host_signing_enabled: bool = False
    host_signing_public_key_id: str = "attestation-demo-host-1"


class ApplicationSecurityProfile(BaseModel):
    """Per-application V-SEC toggles."""

    model_config = ConfigDict(extra="forbid")

    prompt_defense_enabled: bool = True
    tool_injection_defense_enabled: bool = True
    retrieval_poisoning_defense_enabled: bool = True
    tenant_security_verify_enabled: bool = True
    immutable_audit_trail_enabled: bool = False
    audit_trail_regions: list[str] = Field(
        default_factory=lambda: ["eu-central-1", "us-east-1"],
    )
    defense_plugin_ids: list[str] = Field(default_factory=list)
    defense_bundle_ids: list[str] = Field(default_factory=list)
    encryption_enforcement_enabled: bool = False
    require_secrets_store_for_encryption: bool = False


class EvaluationProfile(BaseModel):
    """Evaluation and benchmarking posture."""

    model_config = ConfigDict(extra="forbid")

    shadow_eval_enabled: bool = True
    online_registry_enabled: bool = True
    offline_eval_runner_enabled: bool = False
    trend_comparison_enabled: bool = True
    require_baseline_for_release: bool = False
    registry_path: Path | None = None
    evaluation_assets_ref: str | None = None


class CriticVerificationScopes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_partial: bool = False
    graph_final: bool = True


class CriticProfile(BaseModel):
    """Critic & Verification Layer posture."""

    model_config = ConfigDict(extra="forbid")

    semantic_judge_enabled: bool = False
    trajectory_eval_enabled: bool = False
    judge_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    require_critic_on_completion: bool = False
    evaluator_loop_max_iterations: int = Field(default=2, ge=1, le=16)
    critic_llm_profile_ref: str | None = None
    critic_llm_profile: LLMProfile | None = None
    default_rubric_ref: str | None = None
    l2_human_required: bool = False
    l2_borderline_margin: float = Field(default=0.05, ge=0.0, le=0.5)
    scopes: CriticVerificationScopes = Field(default_factory=CriticVerificationScopes)


class AdaptiveProfile(BaseModel):
    """Adaptive harness intelligence configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: AdaptiveMode = "observe"
    enabled_loops: list[AdaptiveLoopKind] = Field(default_factory=list)
    utility_weights: UtilityWeights = Field(default_factory=UtilityWeights)
    canary_tenant_allowlist: list[str] = Field(default_factory=list)
    canary_traffic_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    human_approver_group: str | None = None
    profile_versions_db_path: Path | None = None
    profile_pointers_db_path: Path | None = None
    signal_store_path: Path | None = None
    proposal_store_path: Path | None = None
    debug_readonly_routes: bool = False
    feature_flag_slug: str | None = None
    rollout_flag_key: str = "harness.adaptive.recommend"
    live_model_routing_enabled: bool = False
    capability_marketplace_enabled: bool = False


class OrchestrationProfile(BaseModel):
    """Nexus loop composition overrides."""

    model_config = ConfigDict(extra="forbid")

    planner_kind: str | None = None
    classifier_kind: str | None = None
    retry_policy_name: str | None = None
    long_running_enabled: bool = False
    max_delegation_depth: int = Field(default=4, ge=1, le=32)
    max_parallel_nodes: int | None = Field(default=None, ge=1, le=256)
    max_inflight_nodes: int | None = Field(default=None, ge=1, le=256)
    max_run_retries: int = Field(default=0, ge=0, le=32)
    merge_strategy: str = "concat"
    multi_agent_order: str = "registry"
    allow_dynamic_replan: bool = False
    intent_routes: list[IntentRoute] = Field(default_factory=list)
    coordination_pattern: str | None = None
    emit_coordination_advisory: bool = False
