# © Artur Czarnecki. All rights reserved.

"""Typed Tier-3 environment sub-profiles (Phase H-APP.1.1 · APP-EVOL-8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.applications.contracts.agent_governance import AgentGovernanceProfile
from intergrax.applications.contracts.application_recovery_contract import ApplicationRecoveryContract
from intergrax.applications.contracts.business_outcome_webhook import BusinessOutcomeWebhookConfig
from intergrax.applications.contracts.capability_alias import CapabilityGovernanceProfile
from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.agent_budget import BudgetReactionProfile
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.resilience_policy import ResiliencePolicy, default_resilience_policy
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.adaptive.contracts import UtilityWeights
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
from intergrax.runtime.capacity.contracts import ScalingPolicy
from intergrax.runtime.events.event_taxonomy import EventCategory
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.policy.compliance_profiles import ComplianceDomainClass


class IdentityProfile(BaseModel):
    """Harness identity posture for a Tier-3 host (Phase H-APP.2.1)."""

    model_config = ConfigDict(extra="forbid")

    require_api_key: bool = False
    api_key_env: str = "INTERGRAX_HARNESS_API_KEY"
    tenant_required: bool = False
    role_claims_header: str | None = None
    service_identities: dict[str, str] = Field(default_factory=dict)
    critical_action_signing_enabled: bool = False
    critical_action_signing_secret_env: str = "INTERGRAX_CRITICAL_ACTION_SIGNING_KEY"


class PolicyRulesProfile(BaseModel):
    """Declarative policy rules file reference (Phase H-APP.2.4)."""

    model_config = ConfigDict(extra="forbid")

    rules_path: Path | None = None
    inline_rules: list[dict[str, Any]] = Field(default_factory=list)


class ApplicationSecurityProfile(BaseModel):
    """Per-application V-SEC toggles (Phase H-APP.2.7)."""

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


class GuardrailProfile(BaseModel):
    """Vendor LLM guardrail scanning toggles (M-P12-WIRE.1)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    scan_input: bool = True
    scan_output: bool = True
    scan_tool_calls: bool = False
    secondary_slug: str | None = None
    colang_config_path: str | None = None
    bedrock_guardrail_policy_id: str | None = None
    inference_slug: str | None = None


ContextEnginePreset = Literal[
    "default", "codebase", "regulated_minimal", "explore_child", "custom"
]


class ContextDecisionProfile(BaseModel):
    """Unified memory vs context vs RAG assembly policy (Phase MEM-CTX.1)."""

    model_config = ConfigDict(extra="forbid")

    include_session_history: bool = True
    prefer_longterm_memory: bool = True
    prefer_rag_when_enabled: bool = True
    max_memory_entries_in_context: int = Field(default=8, ge=1, le=64)


class PromptProfile(BaseModel):
    """YAML prompt catalog selection for a Tier-3 host (Phase PE-1)."""

    model_config = ConfigDict(extra="forbid")

    catalog_path: Path | None = None
    load_on_startup: bool = True
    approval_required: bool = False


class ContextProfile(BaseModel):
    """Context assembly defaults for Nexus (Phase H-APP.4.1, CE-2.6)."""

    model_config = ConfigDict(extra="forbid")

    assembly_options: TaskContextAssemblyOptions = Field(
        default_factory=TaskContextAssemblyOptions
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
    default_history_compression: Literal["truncate_oldest", "summarize_oldest", "hybrid"] = "truncate_oldest"

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
    """Task/org memory flags (Phase H-APP.4.2)."""

    model_config = ConfigDict(extra="forbid")

    enable_user_memory: bool = False
    enable_org_memory: bool = False
    enable_long_term_memory: bool = False
    enable_task_memory: bool = False
    retention_days: int | None = Field(default=None, ge=1)
    scope_boundary: str = "tenant"
    consolidation_mode: Literal["manual", "scheduled", "auto"] = "manual"
    enable_entity_graph_memory: bool = False
    enable_session_vector_index: bool = False
    include_cross_session_episodic: bool = False
    session_index_top_k: int = Field(default=8, ge=1)
    session_index_score_threshold: float | None = None
    vector_index_namespace: str | None = None
    session_index_roles: tuple[str, ...] = ("user", "assistant")


class ReliabilityProfile(BaseModel):
    """Idempotency, circuit breaker, checkpoint, scheduler (Phase H-APP.4.5)."""

    model_config = ConfigDict(extra="forbid")

    idempotency_enabled: bool = True
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    checkpoint_interval_steps: int = Field(default=1, ge=1)
    long_running_scheduler_enabled: bool = False
    resilience_policy: ResiliencePolicy = Field(default_factory=default_resilience_policy)
    default_autonomy_level: AutonomyLevel = AutonomyLevel.ASK
    tenant_autonomy_ceiling: AutonomyLevel | None = None
    compensation_enabled: bool = False
    partial_results_enabled: bool = False
    middleware_hook_timeout_seconds: float = Field(default=2.0, ge=0.01, le=60.0)
    recovery_contract: ApplicationRecoveryContract | None = None


class EventSubscriptionSpec(BaseModel):
    """Declarative runtime bus subscription for a Tier-3 host (OBS-EVOL-9.10)."""

    model_config = ConfigDict(extra="forbid")

    subscription_id: str = Field(min_length=1)
    handler_id: str = Field(min_length=1)
    kind_prefix: str | None = None
    categories: list[EventCategory] | None = None
    ops_hints: list[str] | None = None
    event_types: list[RuntimeEventType] | None = None
    priority: int = Field(default=100, ge=0, le=1000)
    enabled: bool = True

    @field_validator("kind_prefix")
    @classmethod
    def _strip_kind_prefix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("ops_hints")
    @classmethod
    def _strip_ops_hints(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned or None

    def has_filter(self) -> bool:
        return bool(
            self.kind_prefix
            or self.categories
            or self.ops_hints
            or self.event_types
        )


class ObservabilityProfile(BaseModel):
    """Trace, OTEL, metrics, optional product debug (Phase H-APP.4.8)."""

    model_config = ConfigDict(extra="forbid")

    trace_sqlite_enabled: bool = True
    otel_enabled: bool = False
    metrics_plugins_enabled: bool = True
    debug_surface_override: bool | None = None
    causal_diagnostics_enabled: bool = False
    health_dashboard_enabled: bool = False
    unified_observability_dashboard_enabled: bool = False
    event_subscriptions: list[EventSubscriptionSpec] = Field(default_factory=list)

    @field_validator("event_subscriptions")
    @classmethod
    def _validate_event_subscriptions(
        cls,
        specs: list[EventSubscriptionSpec],
    ) -> list[EventSubscriptionSpec]:
        seen: set[str] = set()
        for spec in specs:
            if spec.subscription_id in seen:
                raise ValueError(
                    f"duplicate event subscription_id: {spec.subscription_id!r}"
                )
            seen.add(spec.subscription_id)
            if spec.enabled and not spec.has_filter():
                raise ValueError(
                    f"subscription {spec.subscription_id!r} requires at least one filter "
                    "(kind_prefix, categories, ops_hints, or event_types)"
                )
        return specs


class CostProfile(BaseModel):
    """Run budget and quota governance for a Tier-3 host (Phase COST-1)."""

    model_config = ConfigDict(extra="forbid")

    budget_enforcement_enabled: bool = True
    enforcement_mode: Literal["abort", "hitl"] = "abort"
    budget_reaction: BudgetReactionProfile | None = None
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_llm_calls: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    max_planner_iterations: int | None = Field(default=None, ge=1)
    quota_degrade_threshold_ratio: float = Field(default=0.90, ge=0.0, le=1.0)
    forecasting_enabled: bool = False
    optimization_recommendations_enabled: bool = False
    tenant_fairness_quotas_enabled: bool = False


class ComplianceProfile(BaseModel):
    """Regulated-domain compliance template selection (AUDIT-IDEAL-5.2)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    domain_class: ComplianceDomainClass = ComplianceDomainClass.REGULATED


class EvaluationProfile(BaseModel):
    """Evaluation and benchmarking posture for a Tier-3 host (Phase EVAL-1)."""

    model_config = ConfigDict(extra="forbid")

    shadow_eval_enabled: bool = True
    online_registry_enabled: bool = True
    offline_eval_runner_enabled: bool = False
    trend_comparison_enabled: bool = True
    require_baseline_for_release: bool = False
    registry_path: Path | None = None
    evaluation_assets_ref: str | None = None


class CriticVerificationScopes(BaseModel):
    """Which execution scopes run CVL checks when semantic/trajectory critics are enabled."""

    model_config = ConfigDict(extra="forbid")

    node_partial: bool = False
    graph_final: bool = True
    uaep_step: bool = False


class CriticProfile(BaseModel):
    """Critic & Verification Layer posture for a Tier-3 host (Phase CRIT-V-1.1)."""

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


AdaptiveMode = Literal["observe", "recommend", "shadow", "canary", "apply"]


class AdaptiveProfile(BaseModel):
    """Adaptive Harness Intelligence Tier-3 configuration (AHIA Â§14.5, W-ADAPT-4.1)."""

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
    business_outcome_webhook: BusinessOutcomeWebhookConfig | None = None
    feature_flag_slug: str | None = None
    rollout_flag_key: str = "harness.adaptive.recommend"
    live_model_routing_enabled: bool = False
    capability_marketplace_enabled: bool = False


class GovernanceProfile(BaseModel):
    """Platform governance cadence for Tier-3 hosts (AUDIT-IDEAL-1.1 / 1.2)."""

    model_config = ConfigDict(extra="forbid")

    quarterly_strategy_review_enabled: bool = False
    architecture_health_metrics_enabled: bool = False
    governance_dashboard_enabled: bool = False


class IntegrationGovernanceProfile(BaseModel):
    """Integration marketplace and catalog governance (AUDIT-IDEAL-13.1 / 13.2)."""

    model_config = ConfigDict(extra="forbid")

    marketplace_catalog_enabled: bool = False
    catalog_hot_reload_enabled: bool = False


class ScalingProfile(BaseModel):
    """Elastic capacity posture (ECP-1.1)."""

    model_config = ConfigDict(extra="forbid")

    policy: ScalingPolicy = Field(default_factory=ScalingPolicy)
    production_adapters_enabled: bool = False


class HostDeploymentProfile(BaseModel):
    """Product host deployment modes (AUDIT-IDEAL-28.3 / 28.4)."""

    model_config = ConfigDict(extra="forbid")

    lkw_hybrid_daemon_enabled: bool = False
    lkw_daemon_bind_host: str = "127.0.0.1"
    lkw_daemon_port: int = Field(default=8020, ge=1, le=65535)
    business_agents_deploy_enabled: bool = False


class OrchestrationProfile(BaseModel):
    """Nexus loop composition overrides (Phase H-APP.3.1)."""

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


class ShadowWorkspaceProfile(BaseModel):
    """Shadow workspace paths and retention (Phase H-APP.3.4)."""

    model_config = ConfigDict(extra="forbid")

    root: Path | None = None
    retention_hours: int | None = Field(default=None, ge=1)


class ExecutionBoundaryExportProfile(BaseModel):
    """Execution Boundary Export (EBE) â€” unsigned tool-boundary events for external attestation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    capture_mode: str = Field(
        default="side_effects_only",
        description="off | side_effects_only | allowlist",
    )
    allowlist: list[str] = Field(default_factory=list)
    include_canonical_io: bool = True
    step_level_enabled: bool = Field(
        default=False,
        description="Emit harness_step events from HarnessKernel.execute_step (EBE-8).",
    )
    host_signing_enabled: bool = Field(
        default=False,
        description="Ed25519 host-attestation envelope per boundary event (EBE-9).",
    )
    host_signing_public_key_id: str = Field(
        default="attestation-demo-host-1",
        description="Pinned public key id surfaced in host_attestation envelopes.",
    )


class SandboxProfile(BaseModel):
    """Sandbox session manager configuration (Phase H-APP.3.5)."""

    model_config = ConfigDict(extra="forbid")

    root: Path | None = None
    enable_exec_tool: bool = False


class ToolSelectionConfig(BaseModel):
    """Tool catalog selection posture (APP-EVOL-8 Â· CapabilityBundle)."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "static"
    top_k: int = Field(default=20, ge=1, le=100)


class ToolInvocationConfig(BaseModel):
    """Tool invocation loop posture (APP-EVOL-8 Â· CapabilityBundle)."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "single_pass"
    max_parallel: int = Field(default=8, ge=1, le=32)

