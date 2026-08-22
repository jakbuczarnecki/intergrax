# © Artur Czarnecki. All rights reserved.

"""Nested profile bundle containers (APP-EVOL-8 · ADR-APP-003)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.applications.contracts.application_host import ApplicationFeatures, ApplicationProfile
from intergrax.applications.contracts.capability_alias import CapabilityGovernanceProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.persistence_topology import (
    DeploymentTopology,
    PersistenceTopology,
    required_persistence_for_deployment,
)
from intergrax.applications.contracts.agent_governance import AgentGovernanceProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.reasoning_profile import ReasoningProfile
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import LLMRoutingProfile
from intergrax.runtime.modality.modality_profile import ModalityProfile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile

from intergrax.applications.contracts.environment_profile.domain_policy import DomainPolicyFragments
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    AdaptiveProfile,
    ApplicationSecurityProfile,
    ComplianceProfile,
    ContextProfile,
    CostProfile,
    CriticProfile,
    CriticVerificationScopes,
    EvaluationProfile,
    ExecutionBoundaryExportProfile,
    GovernanceProfile,
    GuardrailProfile,
    HostDeploymentProfile,
    IdentityProfile,
    IntegrationGovernanceProfile,
    MemoryProfile,
    ObservabilityProfile,
    OrchestrationProfile,
    PolicyRulesProfile,
    PromptProfile,
    ReliabilityProfile,
    SandboxProfile,
    ScalingProfile,
    ShadowWorkspaceProfile,
    ToolInvocationConfig,
    ToolSelectionConfig,
)

if TYPE_CHECKING:
    from intergrax.applications.contracts.org_policy import OrganizationalPolicyEnvelope


class HostMeta(BaseModel):
    """Host identity posture bundle."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = "default"
    spec_version: str = "1.0.0"
    application_profile: ApplicationProfile = ApplicationProfile.LAB
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    deployment_topology: DeploymentTopology = DeploymentTopology.PROCESS_LOCAL
    features: ApplicationFeatures = Field(default_factory=ApplicationFeatures.lab_defaults)

    @model_validator(mode="before")
    @classmethod
    def _reject_contradictory_persistence_requirement(cls, data: object) -> object:
        if not isinstance(data, dict) or "required_persistence_topology" not in data:
            return data
        payload = dict(data)
        declared = payload.pop("required_persistence_topology")
        required = required_persistence_for_deployment(
            payload.get("deployment_topology", DeploymentTopology.PROCESS_LOCAL),
        )
        declared_topology = (
            declared
            if isinstance(declared, PersistenceTopology)
            else PersistenceTopology(declared)
        )
        if declared_topology != required:
            raise ValueError(
                "required_persistence_topology contradicts deployment_topology",
            )
        return payload

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> HostMeta:
        if update is not None and "required_persistence_topology" in update:
            payload = self.model_dump()
            payload.update(dict(update))
            return type(self).model_validate(payload)
        return super().model_copy(update=update, deep=deep)

    @property
    def required_persistence_topology(self) -> PersistenceTopology:
        """Persistence capability required by ``deployment_topology`` (derived)."""
        return required_persistence_for_deployment(self.deployment_topology)

    @classmethod
    def lab(cls, *, profile_id: str = "lab.default") -> HostMeta:
        return cls(
            profile_id=profile_id,
            application_profile=ApplicationProfile.LAB,
            execution_mode=ExecutionMode.BALANCED,
            deployment_topology=DeploymentTopology.PROCESS_LOCAL,
            features=ApplicationFeatures.lab_defaults(),
        )

    @classmethod
    def product(
        cls,
        *,
        profile_id: str = "product.default",
        deployment_topology: DeploymentTopology = DeploymentTopology.SINGLE_HOST,
    ) -> HostMeta:
        return cls(
            profile_id=profile_id,
            application_profile=ApplicationProfile.PRODUCT,
            execution_mode=ExecutionMode.STRICT,
            deployment_topology=deployment_topology,
            features=ApplicationFeatures.product_defaults(),
        )


class SecurityEnvelope(BaseModel):
    """Trust boundary and org policy bundle."""

    model_config = ConfigDict(extra="forbid")

    identity: IdentityProfile = Field(default_factory=IdentityProfile)
    application_security: ApplicationSecurityProfile = Field(
        default_factory=ApplicationSecurityProfile,
    )
    guardrails: GuardrailProfile = Field(default_factory=GuardrailProfile)
    policy_rules: PolicyRulesProfile | None = None
    compliance: ComplianceProfile = Field(default_factory=ComplianceProfile)
    organizational_policy: OrganizationalPolicyEnvelope | None = None

    @classmethod
    def lab(cls) -> SecurityEnvelope:
        return cls(identity=IdentityProfile(require_api_key=False))

    @classmethod
    def strict(
        cls,
        *,
        org: OrganizationalPolicyEnvelope | None = None,
    ) -> SecurityEnvelope:
        return cls(
            identity=IdentityProfile(
                require_api_key=True,
                tenant_required=True,
                critical_action_signing_enabled=True,
            ),
            application_security=ApplicationSecurityProfile(
                prompt_defense_enabled=True,
                tool_injection_defense_enabled=True,
                retrieval_poisoning_defense_enabled=True,
                tenant_security_verify_enabled=True,
                immutable_audit_trail_enabled=True,
                defense_bundle_ids=["harness.strict_injection"],
            ),
            compliance=ComplianceProfile(enabled=True),
            organizational_policy=org,
        )

    @classmethod
    def production(
        cls,
        *,
        org: OrganizationalPolicyEnvelope | None = None,
    ) -> SecurityEnvelope:
        """Production preset composing S1+S2+S3 security toggles (Phase SEC-BUNDLE-2)."""
        base = cls.strict(org=org)
        return base.model_copy(
            update={
                "application_security": base.application_security.model_copy(
                    update={
                        "encryption_enforcement_enabled": True,
                        "require_secrets_store_for_encryption": True,
                    },
                ),
            },
        )


class CapabilityBundle(BaseModel):
    """Tier-0 catalogs and context planes."""

    model_config = ConfigDict(extra="forbid")

    integrations: IntegrationProfile = Field(default_factory=IntegrationProfile.lab)
    tools: ToolProfile = Field(default_factory=ToolProfile)
    skills: SkillProfile = Field(default_factory=SkillProfile)
    llm: LLMProfile | None = None
    llm_routing: LLMRoutingProfile | None = None
    llm_routing_evaluating_secondary: bool = False
    modality: ModalityProfile | None = None
    prompt: PromptProfile = Field(default_factory=PromptProfile)
    context: ContextProfile = Field(default_factory=ContextProfile)
    memory: MemoryProfile = Field(default_factory=MemoryProfile)
    tool_selection: ToolSelectionConfig = Field(default_factory=ToolSelectionConfig)
    tool_invocation: ToolInvocationConfig = Field(default_factory=ToolInvocationConfig)

    @classmethod
    def lab(
        cls,
        *,
        tool_enabled: list[str],
        skill_profile: SkillProfile,
        integration_profile: IntegrationProfile,
        modality_profile: ModalityProfile,
        memory_profile: MemoryProfile,
    ) -> CapabilityBundle:
        return cls(
            integrations=integration_profile,
            tools=ToolProfile(enabled=tool_enabled),
            skills=skill_profile,
            llm=LLMProfile.lab(),
            modality=modality_profile,
            context=ContextProfile(enable_rag=True, enable_websearch=True),
            memory=memory_profile,
            tool_selection=ToolSelectionConfig(mode="skill_pack"),
        )


class CognitionBundle(BaseModel):
    """Reasoning, orchestration, critic, and adaptive loops."""

    model_config = ConfigDict(extra="forbid")

    reasoning: ReasoningProfile = Field(default_factory=ReasoningProfile)
    orchestration: OrchestrationProfile = Field(default_factory=OrchestrationProfile)
    critic: CriticProfile = Field(default_factory=CriticProfile)
    adaptive: AdaptiveProfile = Field(default_factory=AdaptiveProfile)
    evaluation: EvaluationProfile = Field(default_factory=EvaluationProfile)
    codecraft: CodeCraftProfile | None = None

    @classmethod
    def lab(cls, *, long_running: bool = True) -> CognitionBundle:
        return cls(
            orchestration=OrchestrationProfile(long_running_enabled=long_running),
            adaptive=AdaptiveProfile(enabled=False, mode="observe"),
            evaluation=EvaluationProfile(
                shadow_eval_enabled=True,
                online_registry_enabled=True,
                offline_eval_runner_enabled=True,
                trend_comparison_enabled=True,
            ),
            codecraft=CodeCraftProfile(mode="supervised", isolation_tier="local"),
        )

    @classmethod
    def regulated(cls) -> CognitionBundle:
        return cls(
            critic=CriticProfile(
                semantic_judge_enabled=True,
                require_critic_on_completion=True,
                scopes=CriticVerificationScopes(graph_final=True),
            ),
            evaluation=EvaluationProfile(require_baseline_for_release=True),
        )


class GovernanceBundle(BaseModel):
    """Reliability, observability, cost, and platform ops."""

    model_config = ConfigDict(extra="forbid")

    reliability: ReliabilityProfile = Field(default_factory=ReliabilityProfile)
    observability: ObservabilityProfile = Field(default_factory=ObservabilityProfile)
    cost: CostProfile = Field(default_factory=CostProfile)
    scaling: ScalingProfile = Field(default_factory=ScalingProfile)
    platform: GovernanceProfile = Field(default_factory=GovernanceProfile)
    capability: CapabilityGovernanceProfile = Field(default_factory=CapabilityGovernanceProfile)
    agent: AgentGovernanceProfile = Field(default_factory=AgentGovernanceProfile)
    integration_marketplace: IntegrationGovernanceProfile = Field(
        default_factory=IntegrationGovernanceProfile,
    )
    deployment: HostDeploymentProfile = Field(default_factory=HostDeploymentProfile)
    boundary_export: ExecutionBoundaryExportProfile | None = None

    @classmethod
    def lab(cls) -> GovernanceBundle:
        return cls(
            reliability=ReliabilityProfile(
                long_running_scheduler_enabled=True,
                idempotency_enabled=True,
                partial_results_enabled=True,
            ),
            observability=ObservabilityProfile(
                trace_sqlite_enabled=True,
                otel_enabled=False,
                metrics_plugins_enabled=True,
                debug_surface_override=True,
            ),
            cost=CostProfile(max_llm_calls=64, max_tool_calls=128),
        )

    @classmethod
    def production_slo(cls) -> GovernanceBundle:
        return cls(
            observability=ObservabilityProfile(
                trace_sqlite_enabled=True,
                otel_enabled=True,
                metrics_plugins_enabled=True,
                debug_surface_override=False,
                causal_diagnostics_enabled=True,
                health_dashboard_enabled=True,
                unified_observability_dashboard_enabled=True,
            ),
            platform=GovernanceProfile(
                quarterly_strategy_review_enabled=True,
                architecture_health_metrics_enabled=True,
                governance_dashboard_enabled=True,
            ),
        )


class TopologyBundle(BaseModel):
    """Declarative multi-agent topology."""

    model_config = ConfigDict(extra="forbid")

    graph_spec: ApplicationGraphSpec | None = None


class IsolationBundle(BaseModel):
    """Shadow workspace and sandbox isolation."""

    model_config = ConfigDict(extra="forbid")

    shadow_workspace: ShadowWorkspaceProfile | None = None
    sandbox: SandboxProfile | None = None

    @classmethod
    def lab(cls) -> IsolationBundle:
        return cls(
            shadow_workspace=ShadowWorkspaceProfile(),
            sandbox=SandboxProfile(enable_exec_tool=True),
        )

    @classmethod
    def product(cls) -> IsolationBundle:
        return cls(sandbox=SandboxProfile(enable_exec_tool=True))


class EnvironmentExtensions(BaseModel):
    """Typed escape hatch for product-specific policy slices."""

    model_config = ConfigDict(extra="forbid")

    domain_policy_fragments: DomainPolicyFragments = Field(
        default_factory=DomainPolicyFragments,
    )
