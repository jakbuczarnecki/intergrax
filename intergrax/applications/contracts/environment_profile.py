# © Artur Czarnecki. All rights reserved.

"""IDEAL §17 umbrella — typed Tier-3 application environment (Phase H-APP.1.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.reasoning_profile import ReasoningProfile
from intergrax.contracts.resilience_policy import ResiliencePolicy, default_resilience_policy
from intergrax.runtime.capacity.contracts import ScalingPolicy
from intergrax.applications.contracts.agent_governance import AgentGovernanceProfile
from intergrax.applications.contracts.application_recovery_contract import (
    ApplicationRecoveryContract,
    standard_strict_product_recovery_contract,
)
from intergrax.applications.contracts.capability_alias import CapabilityGovernanceProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.applications.contracts.intent_route import IntentRoute
from intergrax.applications.contracts.application_host import ApplicationFeatures, ApplicationProfile
from intergrax.contracts.agent_budget import BudgetReactionProfile
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.modality.modality_profile import ModalityProfile, lab_default_modality_profile
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile
from intergrax.applications.contracts.business_outcome_webhook import BusinessOutcomeWebhookConfig
from intergrax.runtime.adaptive.contracts import UtilityWeights
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
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
    """Context assembly defaults for Nexus (Phase H-APP.4.1)."""

    model_config = ConfigDict(extra="forbid")

    assembly_options: TaskContextAssemblyOptions = Field(
        default_factory=TaskContextAssemblyOptions
    )
    budget_policy: ContextBudgetPolicy | None = None
    decision: ContextDecisionProfile = Field(default_factory=ContextDecisionProfile)
    enable_rag: bool = True
    enable_websearch: bool = True
    drift_monitoring_enabled: bool = False
    drift_alert_threshold: float = Field(default=0.35, ge=0.0, le=2.0)
    semantic_compression_enabled: bool = False
    default_history_compression: Literal["truncate_oldest", "summarize_oldest", "hybrid"] = "truncate_oldest"


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
    """Adaptive Harness Intelligence Tier-3 configuration (AHIA §14.5, W-ADAPT-4.1)."""

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


class SandboxProfile(BaseModel):
    """Sandbox session manager configuration (Phase H-APP.3.5)."""

    model_config = ConfigDict(extra="forbid")

    root: Path | None = None
    enable_exec_tool: bool = False


class ApplicationEnvironmentProfile(BaseModel):
    """
    Aggregated Tier-3 environment — single composition contract (IDEAL §17).

    Wired via :func:`~intergrax.applications._shared.environment_wiring.wire_application_environment`.
    """

    model_config = ConfigDict(extra="forbid")

    profile_id: str = "default"
    spec_version: str = Field(
        default="1.0.0",
        description="Serialized environment spec version for UI round-trip (Phase DX-7.2)",
    )
    application_profile: ApplicationProfile = ApplicationProfile.LAB
    integration_profile: IntegrationProfile = Field(default_factory=IntegrationProfile.lab)
    tool_profile: ToolProfile = Field(default_factory=ToolProfile)
    skill_profile: SkillProfile = Field(default_factory=SkillProfile)
    modality_profile: ModalityProfile | None = None
    llm_profile: LLMProfile | None = None
    prompt_profile: PromptProfile = Field(default_factory=PromptProfile)
    context_profile: ContextProfile = Field(default_factory=ContextProfile)
    memory_profile: MemoryProfile = Field(default_factory=MemoryProfile)
    reliability_profile: ReliabilityProfile = Field(default_factory=ReliabilityProfile)
    observability_profile: ObservabilityProfile = Field(default_factory=ObservabilityProfile)
    cost_profile: CostProfile = Field(default_factory=CostProfile)
    compliance_profile: ComplianceProfile = Field(default_factory=ComplianceProfile)
    evaluation_profile: EvaluationProfile = Field(default_factory=EvaluationProfile)
    critic_profile: CriticProfile = Field(default_factory=CriticProfile)
    adaptive_profile: AdaptiveProfile = Field(default_factory=AdaptiveProfile)
    orchestration_profile: OrchestrationProfile = Field(default_factory=OrchestrationProfile)
    reasoning_profile: ReasoningProfile = Field(default_factory=ReasoningProfile)
    scaling_profile: ScalingProfile = Field(default_factory=ScalingProfile)
    governance_profile: GovernanceProfile = Field(default_factory=GovernanceProfile)
    capability_governance_profile: CapabilityGovernanceProfile = Field(
        default_factory=CapabilityGovernanceProfile,
    )
    agent_governance_profile: AgentGovernanceProfile = Field(
        default_factory=AgentGovernanceProfile,
    )
    host_deployment_profile: HostDeploymentProfile = Field(default_factory=HostDeploymentProfile)
    integration_governance_profile: IntegrationGovernanceProfile = Field(
        default_factory=IntegrationGovernanceProfile
    )
    identity_profile: IdentityProfile = Field(default_factory=IdentityProfile)
    security_profile: ApplicationSecurityProfile = Field(
        default_factory=ApplicationSecurityProfile
    )
    guardrail_profile: GuardrailProfile = Field(default_factory=GuardrailProfile)
    policy_rules: PolicyRulesProfile | None = None
    organizational_policy: "OrganizationalPolicyEnvelope | None" = None
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    graph_spec: ApplicationGraphSpec | None = None
    shadow_workspace: ShadowWorkspaceProfile | None = None
    sandbox: SandboxProfile | None = None
    features: ApplicationFeatures = Field(default_factory=ApplicationFeatures.lab_defaults)
    domain_policy_fragments: dict[str, Any] = Field(default_factory=dict)
    tool_selection_mode: str = "static"
    tool_selection_top_k: int = Field(default=20, ge=1, le=100)

    @classmethod
    def harness_memory_profile(cls) -> MemoryProfile:
        """STM/LTM/task memory flags for harness reference hosts (Phase MEM)."""
        return MemoryProfile(
            enable_user_memory=True,
            enable_org_memory=True,
            enable_long_term_memory=True,
            enable_task_memory=True,
        )

    def with_harness_memory(self) -> ApplicationEnvironmentProfile:
        """Return a copy with harness memory flags enabled (sqlite-backed hosts)."""
        return self.model_copy(update={"memory_profile": self.harness_memory_profile()})

    @classmethod
    def lab_org_virtual_workforce_defaults(
        cls,
        *,
        profile_id: str = "lab.org.virtual_workforce",
    ) -> ApplicationEnvironmentProfile:
        """UC-11 reference host — strict organizational envelope (ACP-ORG-5)."""
        from intergrax.applications.contracts.org_policy import lab_strict_org_envelope

        return cls.lab_defaults(profile_id=profile_id).with_uc11_organizational_policy(
            lab_strict_org_envelope(),
        )

    def with_uc11_organizational_policy(
        self,
        envelope: OrganizationalPolicyEnvelope,
    ) -> ApplicationEnvironmentProfile:
        """Attach org envelope and STRICT execution for UC-11 product hosts."""
        return self.model_copy(
            update={
                "execution_mode": ExecutionMode.STRICT,
                "organizational_policy": envelope,
            },
        )

    @classmethod
    def lab_defaults(
        cls,
        *,
        profile_id: str = "lab.default",
        harness_tools: bool = True,
    ) -> ApplicationEnvironmentProfile:
        """Reference lab harness environment preset."""
        from intergrax.applications._shared.skill_wiring import lab_skill_profile

        tool_enabled = ["rag.retrieve", "websearch.query"]
        if harness_tools:
            tool_enabled.extend(
                [
                    "errors.capture",
                    "harness.echo",
                    "harness.skill_registry",
                    "sandbox.exec",
                ]
            )
            tool_enabled.extend(
                [
                    "speech.synthesize",
                    "speech.transcribe",
                    "vision.detect",
                    "vision.segment",
                    "vision.ocr_regions",
                    "ml.predict",
                    "ml.explain",
                    "ml.batch_predict",
                ]
            )
        return cls(
            profile_id=profile_id,
            application_profile=ApplicationProfile.LAB,
            integration_profile=IntegrationProfile.lab_harness_preset(),
            tool_profile=ToolProfile(enabled=tool_enabled),
            skill_profile=lab_skill_profile(),
            tool_selection_mode="skill_pack",
            modality_profile=lab_default_modality_profile(),
            llm_profile=LLMProfile.lab(),
            context_profile=ContextProfile(enable_rag=True, enable_websearch=True),
            memory_profile=cls.harness_memory_profile(),
            reliability_profile=ReliabilityProfile(
                long_running_scheduler_enabled=True,
                idempotency_enabled=True,
                partial_results_enabled=True,
            ),
            observability_profile=ObservabilityProfile(
                trace_sqlite_enabled=True,
                otel_enabled=False,
                metrics_plugins_enabled=True,
                debug_surface_override=True,
            ),
            cost_profile=CostProfile(max_llm_calls=64, max_tool_calls=128),
            evaluation_profile=EvaluationProfile(
                shadow_eval_enabled=True,
                online_registry_enabled=True,
                offline_eval_runner_enabled=True,
                trend_comparison_enabled=True,
            ),
            adaptive_profile=AdaptiveProfile(enabled=False, mode="observe"),
            orchestration_profile=OrchestrationProfile(long_running_enabled=True),
            identity_profile=IdentityProfile(require_api_key=False),
            shadow_workspace=ShadowWorkspaceProfile(),
            sandbox=SandboxProfile(enable_exec_tool=True),
            features=ApplicationFeatures.lab_defaults(),
            execution_mode=ExecutionMode.BALANCED,
        )

    @classmethod
    def harness_production_defaults(
        cls,
        *,
        profile_id: str = "harness.production",
        harness_tools: bool = True,
        secrets_slug: str = "doppler",
        enable_grafana_stack: bool = True,
    ) -> ApplicationEnvironmentProfile:
        """Harness production Tier-3 preset — catalog secrets + observability stack (no business agents)."""
        from intergrax.applications._shared.skill_wiring import lab_skill_profile
        from intergrax.integrations.registry import presets

        base = cls.lab_defaults(profile_id=profile_id, harness_tools=harness_tools)
        return base.model_copy(
            update={
                "application_profile": ApplicationProfile.LAB,
                "integration_profile": presets.harness_production_stack(
                    secrets_slug=secrets_slug,
                    enable_grafana_stack=enable_grafana_stack,
                ),
                "skill_profile": lab_skill_profile(),
                "observability_profile": ObservabilityProfile(
                    trace_sqlite_enabled=True,
                    otel_enabled=True,
                    metrics_plugins_enabled=True,
                    debug_surface_override=False,
                ),
                "adaptive_profile": AdaptiveProfile(
                    enabled=False,
                    mode="observe",
                    feature_flag_slug="unleash",
                    rollout_flag_key="harness.adaptive.recommend",
                ),
                "identity_profile": IdentityProfile(require_api_key=True),
                "execution_mode": ExecutionMode.STRICT,
            }
        )

    @classmethod
    def strict_multi_agent_defaults(
        cls,
        *,
        profile_id: str = "strict.multi_agent",
    ) -> ApplicationEnvironmentProfile:
        """CFG-20 preset — strict execution, structured merge, critic on completion (ORCH-CONFIG.7)."""
        base = cls.lab_defaults(profile_id=profile_id)
        return base.model_copy(
            update={
                "execution_mode": ExecutionMode.STRICT,
                "orchestration_profile": OrchestrationProfile(
                    merge_strategy="structured_json",
                    max_parallel_nodes=8,
                    max_run_retries=1,
                ),
                "critic_profile": CriticProfile(
                    semantic_judge_enabled=True,
                    require_critic_on_completion=True,
                    scopes=CriticVerificationScopes(graph_final=True),
                ),
                "evaluation_profile": EvaluationProfile(
                    shadow_eval_enabled=True,
                    online_registry_enabled=True,
                    offline_eval_runner_enabled=True,
                    require_baseline_for_release=True,
                ),
            }
        )

    def with_reference_host_platform_defaults(
        self,
        *,
        multi_agent_critic: bool = False,
    ) -> ApplicationEnvironmentProfile:
        """CFG-11/13/16 presets for reference Tier-3 hosts (ORCH-CONFIG closeout)."""
        orchestration = self.orchestration_profile.model_copy(
            update={
                "planner_kind": self.orchestration_profile.planner_kind or "engine",
                "classifier_kind": self.orchestration_profile.classifier_kind or "rules",
                "long_running_enabled": True,
            },
        )
        reliability = self.reliability_profile.model_copy(
            update={"long_running_scheduler_enabled": True},
        )
        updates: dict[str, Any] = {
            "orchestration_profile": orchestration,
            "reliability_profile": reliability,
        }
        if multi_agent_critic:
            strict = type(self).strict_multi_agent_defaults(profile_id=self.profile_id)
            updates["execution_mode"] = strict.execution_mode
            updates["critic_profile"] = strict.critic_profile.model_copy(
                update={
                    # CFG-16: require CVL on completion; L1 judge only when rubric is configured.
                    "require_critic_on_completion": True,
                    "semantic_judge_enabled": bool(strict.critic_profile.default_rubric_ref),
                },
            )
            updates["evaluation_profile"] = strict.evaluation_profile
            updates["orchestration_profile"] = orchestration.model_copy(
                update={
                    "merge_strategy": strict.orchestration_profile.merge_strategy,
                    "max_run_retries": strict.orchestration_profile.max_run_retries,
                    "max_parallel_nodes": strict.orchestration_profile.max_parallel_nodes,
                },
            )
        return self.model_copy(update=updates)

    @classmethod
    def async_batch_defaults(
        cls,
        *,
        profile_id: str = "async.batch",
        max_parallel_nodes: int = 8,
    ) -> ApplicationEnvironmentProfile:
        """ORCH-6.2 — deferred execution via queue workers and long-running checkpoints."""
        base = cls.lab_defaults(profile_id=profile_id)
        return base.model_copy(
            update={
                "orchestration_profile": OrchestrationProfile(
                    long_running_enabled=True,
                    merge_strategy="structured_json",
                    max_parallel_nodes=max_parallel_nodes,
                    max_inflight_nodes=max_parallel_nodes,
                ),
                "reliability_profile": ReliabilityProfile(
                    long_running_scheduler_enabled=True,
                    checkpoint_interval_steps=1,
                ),
            }
        )

    @classmethod
    def swarm_exploration_defaults(
        cls,
        *,
        profile_id: str = "swarm.exploration",
        max_parallel_nodes: int = 16,
    ) -> ApplicationEnvironmentProfile:
        """CFG-17 / D7 exploration preset — high parallel cap (ORCH-CONFIG.8 partial)."""
        base = cls.lab_defaults(profile_id=profile_id)
        from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern

        return base.model_copy(
            update={
                "orchestration_profile": OrchestrationProfile(
                    merge_strategy="structured_json",
                    max_parallel_nodes=max_parallel_nodes,
                    max_inflight_nodes=max_parallel_nodes,
                    coordination_pattern=CoordinationPattern.SWARM.value,
                ),
            }
        )

    @classmethod
    def _product_integration_profile(cls) -> IntegrationProfile:
        from intergrax.integrations.registry.presets import POSTGRESQL

        return IntegrationProfile(
            relational_store=POSTGRESQL,
            options={"postgresql": {"tenant_schema": "tenant_default"}},
        )

    @classmethod
    def _product_modality_profile(cls) -> ModalityProfile:
        from intergrax.applications._shared.modality_product_worker_wiring import (
            production_plane_c_modality_profile,
        )

        return production_plane_c_modality_profile()

    @classmethod
    def _product_budget_reaction(cls) -> BudgetReactionProfile:
        from intergrax.applications._shared.budget_wiring import product_budget_reaction

        return product_budget_reaction()

    @classmethod
    def product_defaults(
        cls,
        *,
        profile_id: str = "product.default",
        skill_bundles: list[str] | None = None,
        tool_ids: list[str] | None = None,
        domain_fragments: dict[str, Any] | None = None,
    ) -> ApplicationEnvironmentProfile:
        """Product Tier-3 host preset (legal, research, poc)."""
        bundles = skill_bundles or []
        tools = tool_ids or []
        return cls(
            profile_id=profile_id,
            application_profile=ApplicationProfile.PRODUCT,
            integration_profile=cls._product_integration_profile(),
            tool_profile=ToolProfile(enabled=tools) if tools else ToolProfile(),
            skill_profile=SkillProfile(enabled_bundles=bundles) if bundles else SkillProfile(),
            llm_profile=None,
            context_profile=ContextProfile(
                enable_rag=False,
                enable_websearch=False,
                drift_monitoring_enabled=True,
                semantic_compression_enabled=True,
                default_history_compression="summarize_oldest",
            ),
            memory_profile=MemoryProfile(enable_entity_graph_memory=True),
            reliability_profile=ReliabilityProfile(
                long_running_scheduler_enabled=True,
                compensation_enabled=True,
                partial_results_enabled=True,
                middleware_hook_timeout_seconds=0.25,
                recovery_contract=standard_strict_product_recovery_contract(),
            ),
            observability_profile=ObservabilityProfile(
                trace_sqlite_enabled=True,
                debug_surface_override=False,
                causal_diagnostics_enabled=True,
                health_dashboard_enabled=True,
                unified_observability_dashboard_enabled=True,
            ),
            sandbox=SandboxProfile(enable_exec_tool=True),
            cost_profile=CostProfile(
                max_total_tokens=32_000,
                max_llm_calls=32,
                max_tool_calls=64,
                forecasting_enabled=True,
                optimization_recommendations_enabled=True,
                tenant_fairness_quotas_enabled=True,
                budget_reaction=cls._product_budget_reaction(),
            ),
            compliance_profile=ComplianceProfile(enabled=True),
            prompt_profile=PromptProfile(approval_required=True),
            modality_profile=cls._product_modality_profile(),
            adaptive_profile=AdaptiveProfile(
                enabled=True,
                mode="recommend",
                enabled_loops=[
                    AdaptiveLoopKind.EXECUTION_STRATEGY_TUNING,
                    AdaptiveLoopKind.ROUTING_TUNING,
                ],
                live_model_routing_enabled=True,
                capability_marketplace_enabled=True,
            ),
            integration_governance_profile=IntegrationGovernanceProfile(
                marketplace_catalog_enabled=True,
                catalog_hot_reload_enabled=True,
            ),
            governance_profile=GovernanceProfile(
                quarterly_strategy_review_enabled=True,
                architecture_health_metrics_enabled=True,
                governance_dashboard_enabled=True,
            ),
            host_deployment_profile=HostDeploymentProfile(
                business_agents_deploy_enabled=True,
            ),
            scaling_profile=ScalingProfile(
                policy=ScalingPolicy(enabled=True),
                production_adapters_enabled=True,
            ),
            identity_profile=IdentityProfile(
                require_api_key=True,
                tenant_required=True,
                critical_action_signing_enabled=True,
            ),
            security_profile=ApplicationSecurityProfile(
                prompt_defense_enabled=True,
                tool_injection_defense_enabled=True,
                retrieval_poisoning_defense_enabled=True,
                tenant_security_verify_enabled=True,
                immutable_audit_trail_enabled=True,
                audit_trail_regions=["eu-central-1", "us-east-1"],
            ),
            evaluation_profile=EvaluationProfile(
                shadow_eval_enabled=False,
                online_registry_enabled=True,
                offline_eval_runner_enabled=False,
                require_baseline_for_release=True,
            ),
            orchestration_profile=OrchestrationProfile(long_running_enabled=True),
            features=ApplicationFeatures.product_defaults(),
            execution_mode=ExecutionMode.STRICT,
            domain_policy_fragments=dict(domain_fragments or {}),
        )


from intergrax.applications.contracts.org_policy import OrganizationalPolicyEnvelope  # noqa: E402

ApplicationEnvironmentProfile.model_rebuild()
