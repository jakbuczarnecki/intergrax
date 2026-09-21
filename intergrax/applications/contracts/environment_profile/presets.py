# © Artur Czarnecki. All rights reserved.

"""Declarative ApplicationEnvironmentProfile presets (no composition wiring)."""

from __future__ import annotations

from typing import Any

from intergrax.applications.contracts.application_recovery_contract import (
    standard_strict_product_recovery_contract,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.environment_profile.bundles import (
    CapabilityBundle,
    CognitionBundle,
    EnvironmentExtensions,
    GovernanceBundle,
    HostMeta,
    IsolationBundle,
    SecurityEnvelope,
)
from intergrax.applications.contracts.environment_profile.domain_policy import (
    DomainPolicyFragments,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    AdaptiveProfile,
    ContextProfile,
    CostProfile,
    EvaluationProfile,
    GovernanceProfile,
    HostDeploymentProfile,
    IdentityProfile,
    IntegrationGovernanceProfile,
    MemoryProfile,
    ObservabilityProfile,
    OrchestrationProfile,
    PromptProfile,
    ReliabilityProfile,
    ScalingProfile,
    ToolSelectionConfig,
)
from intergrax.contracts.adaptive_loop_kind import AdaptiveLoopKind
from intergrax.contracts.agent_budget import (
    BudgetExceededReaction,
    BudgetNotifyChannel,
    BudgetReactionProfile,
)
from intergrax.contracts.coordination_pattern import CoordinationPattern
from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationDurabilityMode,
)
from intergrax.contracts.modality_profile import (
    lab_default_modality_profile,
    production_plane_c_modality_profile,
)
from intergrax.contracts.scaling_policy import ScalingPolicy
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.contracts.shipped_manifests import (
    DOCLING,
    GITHUB_ACTIONS,
    GRAFANA,
    LOG,
    LOKI,
    OTEL,
    PGVECTOR,
    POSTGRESQL,
    TEMPO,
    UNLEASH,
)
from intergrax.llm_adapters.contracts.llm_profile import LLMProfile
from intergrax.skills.contracts.skill_profile import SkillProfile
from intergrax.tools.contracts.tool_profile import ToolProfile

_LAB_REFERENCE_TOOL_BUNDLE_IDS: tuple[str, ...] = (
    "catalog",
    "confluence",
    "context",
    "document",
    "harness",
    "health",
    "knowledge",
    "ltm",
    "memory",
    "ml",
    "observability",
    "openai_vector_store",
    "rag",
    "security",
    "skill",
    "storage",
    "vision",
    "websearch",
    "workflow",
    "workspace",
)

_HARNESS_OPTIONAL_TOOL_BUNDLE_IDS: tuple[str, ...] = (
    "sandbox",
    "speech",
)


def harness_memory_profile() -> MemoryProfile:
    return MemoryProfile(
        enable_user_memory=True,
        enable_org_memory=True,
        enable_long_term_memory=True,
        enable_task_memory=True,
    )


def lab_reference_tool_profile(*, harness_tools: bool = True) -> ToolProfile:
    enabled_bundles = list(_LAB_REFERENCE_TOOL_BUNDLE_IDS)
    if harness_tools:
        enabled_bundles.extend(_HARNESS_OPTIONAL_TOOL_BUNDLE_IDS)
    return ToolProfile(enabled_bundles=enabled_bundles)


def lab_skill_profile() -> SkillProfile:
    return SkillProfile(
        enabled_bundles=[
            "harness",
            "legal",
            "research",
            "rag",
            "workspace",
            "memory",
            "knowledge",
        ]
    )


def harness_lab_capability_bundle(*, harness_tools: bool = True) -> CapabilityBundle:
    return CapabilityBundle(
        integrations=IntegrationProfile.lab_harness_preset(),
        tools=lab_reference_tool_profile(harness_tools=harness_tools),
        skills=lab_skill_profile(),
        llm=LLMProfile.lab(),
        modality=lab_default_modality_profile(),
        context=ContextProfile(enable_rag=True, enable_websearch=True),
        memory=harness_memory_profile(),
        tool_selection=ToolSelectionConfig(mode="skill_pack"),
    )


def product_budget_reaction() -> BudgetReactionProfile:
    return BudgetReactionProfile(
        on_agent_limit_exceeded=BudgetExceededReaction.HITL,
        on_environment_limit_exceeded=BudgetExceededReaction.ABORT,
        notify_channels=[BudgetNotifyChannel.TRACE_ONLY, BudgetNotifyChannel.IN_APP],
    )


def product_integration_profile() -> IntegrationProfile:
    return IntegrationProfile(
        relational_store=POSTGRESQL,
        options={"postgresql": {"tenant_schema": "tenant_default"}},
    )


def harness_production_integration_profile(
    *,
    secrets_slug: str = "doppler",
    enable_grafana_stack: bool = True,
) -> IntegrationProfile:
    allowed_secrets = {"doppler", "aws_secrets_manager", "vault"}
    normalized_secrets = secrets_slug.strip().lower()
    if normalized_secrets not in allowed_secrets:
        raise ValueError(
            f"Unsupported secrets slug for harness production stack: {secrets_slug!r}"
        )

    options: dict[str, dict[str, object]] = {OTEL.slug: {}}
    observability_backend = OTEL
    if enable_grafana_stack:
        observability_backend = GRAFANA
        options[LOKI.slug] = {}
        options[TEMPO.slug] = {}
        options[OTEL.slug] = {}

    return IntegrationProfile(
        relational_store=POSTGRESQL,
        vector_store=PGVECTOR,
        notification_channel=LOG,
        document_parser=DOCLING,
        observability_backend=observability_backend,
        secrets_store=normalized_secrets,
        feature_flag=UNLEASH,
        ci_cd=GITHUB_ACTIONS,
        options=options,
    )


def build_lab_defaults(
    *,
    profile_id: str = "lab.default",
    harness_tools: bool = True,
) -> dict[str, Any]:
    return {
        "meta": HostMeta.lab(profile_id=profile_id),
        "security": SecurityEnvelope.lab(),
        "capabilities": harness_lab_capability_bundle(harness_tools=harness_tools),
        "cognition": CognitionBundle.lab(),
        "governance": GovernanceBundle.lab(),
        "isolation": IsolationBundle.lab(),
    }


def build_harness_production_defaults(
    *,
    profile_id: str = "harness.production",
    harness_tools: bool = True,
    secrets_slug: str = "doppler",
    enable_grafana_stack: bool = True,
) -> dict[str, Any]:
    base_caps = harness_lab_capability_bundle(harness_tools=harness_tools)
    return {
        "meta": HostMeta.lab(profile_id=profile_id).model_copy(
            update={"execution_mode": ExecutionMode.STRICT},
        ),
        "security": SecurityEnvelope.lab().model_copy(
            update={"identity": IdentityProfile(require_api_key=True)},
        ),
        "capabilities": base_caps.model_copy(
            update={
                "integrations": harness_production_integration_profile(
                    secrets_slug=secrets_slug,
                    enable_grafana_stack=enable_grafana_stack,
                ),
                "skills": lab_skill_profile(),
            },
        ),
        "cognition": CognitionBundle.lab().model_copy(
            update={
                "adaptive": AdaptiveProfile(
                    enabled=False,
                    mode="observe",
                    feature_flag_slug="unleash",
                    rollout_flag_key="harness.adaptive.recommend",
                ),
            },
        ),
        "governance": GovernanceBundle.lab().model_copy(
            update={
                "observability": ObservabilityProfile(
                    trace_sqlite_enabled=True,
                    otel_enabled=True,
                    metrics_plugins_enabled=True,
                    debug_surface_override=False,
                    bounded_event_delivery_enabled=True,
                ),
            },
        ),
        "isolation": IsolationBundle.lab(),
    }


def build_product_defaults(
    *,
    profile_id: str = "product.default",
    skill_bundles: list[str] | None = None,
    tool_ids: list[str] | None = None,
    domain_fragments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundles = skill_bundles or []
    tools = tool_ids or []
    return {
        "meta": HostMeta.product(profile_id=profile_id),
        "security": SecurityEnvelope.strict(),
        "capabilities": CapabilityBundle(
            integrations=product_integration_profile(),
            tools=ToolProfile(enabled=tools) if tools else ToolProfile(),
            skills=SkillProfile(enabled_bundles=bundles) if bundles else SkillProfile(),
            modality=production_plane_c_modality_profile(),
            prompt=PromptProfile(approval_required=True),
            context=ContextProfile(
                enable_rag=False,
                enable_websearch=False,
                drift_monitoring_enabled=True,
                semantic_compression_enabled=True,
                default_history_compression="summarize_oldest",
            ),
            memory=MemoryProfile(enable_entity_graph_memory=True),
        ),
        "cognition": CognitionBundle(
            orchestration=OrchestrationProfile(
                long_running_enabled=True,
                max_parallel_nodes=8,
                max_inflight_nodes=8,
            ),
            adaptive=AdaptiveProfile(
                enabled=True,
                mode="recommend",
                enabled_loops=[
                    AdaptiveLoopKind.EXECUTION_STRATEGY_TUNING,
                    AdaptiveLoopKind.ROUTING_TUNING,
                ],
                live_model_routing_enabled=True,
                capability_marketplace_enabled=True,
            ),
            evaluation=EvaluationProfile(
                shadow_eval_enabled=False,
                online_registry_enabled=True,
                offline_eval_runner_enabled=False,
                require_baseline_for_release=True,
            ),
        ),
        "governance": GovernanceBundle(
            reliability=ReliabilityProfile(
                long_running_scheduler_enabled=True,
                compensation_enabled=True,
                partial_results_enabled=True,
                middleware_hook_timeout_seconds=0.25,
                recovery_contract=standard_strict_product_recovery_contract(),
                delegated_invocation_correlation_durability=(
                    DelegatedInvocationCorrelationDurabilityMode.REQUIRED
                ),
            ),
            observability=GovernanceBundle.production_slo().observability,
            cost=CostProfile(
                max_total_tokens=32_000,
                max_llm_calls=32,
                max_tool_calls=64,
                forecasting_enabled=True,
                optimization_recommendations_enabled=True,
                tenant_fairness_quotas_enabled=True,
                budget_reaction=product_budget_reaction(),
            ),
            scaling=ScalingProfile(
                policy=ScalingPolicy(enabled=True),
                production_adapters_enabled=True,
            ),
            platform=GovernanceProfile(
                quarterly_strategy_review_enabled=True,
                architecture_health_metrics_enabled=True,
                governance_dashboard_enabled=True,
            ),
            integration_marketplace=IntegrationGovernanceProfile(
                marketplace_catalog_enabled=True,
                catalog_hot_reload_enabled=True,
            ),
            deployment=HostDeploymentProfile(business_agents_deploy_enabled=True),
        ),
        "isolation": IsolationBundle.product(),
        "extensions": EnvironmentExtensions(
            domain_policy_fragments=DomainPolicyFragments.from_runtime_dict(
                domain_fragments,
            ),
        ),
    }


def swarm_coordination_pattern_value() -> str:
    return CoordinationPattern.SWARM.value
