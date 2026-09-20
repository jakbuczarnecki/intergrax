# © Artur Czarnecki. All rights reserved.

"""ApplicationEnvironmentProfile root — nested bundles with flat wire compat (APP-EVOL-8)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.applications.contracts.application_recovery_contract import (
    standard_strict_product_recovery_contract,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.org_policy import OrganizationalPolicyEnvelope
from intergrax.contracts.agent_budget import BudgetReactionProfile
from intergrax.contracts.delegated_invocation_correlation import (
    DelegatedInvocationCorrelationDurabilityMode,
)
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.llm_adapters.contracts.routing_profile import LLMRoutingProfile
from intergrax.contracts.utility_weights import UtilityWeights
from intergrax.contracts.adaptive_loop_kind import AdaptiveLoopKind
from intergrax.contracts.scaling_policy import ScalingPolicy

from intergrax.applications.contracts.environment_profile.bundles import (
    CapabilityBundle,
    CognitionBundle,
    EnvironmentExtensions,
    GovernanceBundle,
    HostMeta,
    IsolationBundle,
    SecurityEnvelope,
    TopologyBundle,
)
from intergrax.applications.contracts.environment_profile.domain_policy import (
    DomainPolicyFragments,
)
from intergrax.applications.contracts.environment_profile.normalization import (
    BUNDLE_ROOT_KEYS,
    PROFILE_SPEC_V2,
    flatten_profile_dict,
    lift_flat_profile_dict,
    migrate_environment_profile_wire,
    uses_nested_profile_wire,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    AdaptiveProfile,
    ApplicationSecurityProfile,
    ComplianceProfile,
    ContextProfile,
    CostProfile,
    DecisionProfile,
    DecisionFlowProfile,
    DecisionPluginProfile,
    DecisionVerificationProfile,
    DiagnosticProfile,
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
)


class ApplicationEnvironmentProfile(BaseModel):
    """
    Aggregated Tier-3 environment — single composition contract (IDEAL §17).

    Nested bundles (§22.6) are canonical storage; flat fields are wire-compatible
    accessors for ``spec_version`` 1.x (ADR-APP-003).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    meta: HostMeta = Field(default_factory=HostMeta)
    security: SecurityEnvelope = Field(default_factory=SecurityEnvelope)
    capabilities: CapabilityBundle = Field(default_factory=CapabilityBundle)
    cognition: CognitionBundle = Field(default_factory=CognitionBundle)
    governance: GovernanceBundle = Field(default_factory=GovernanceBundle)
    topology: TopologyBundle = Field(default_factory=TopologyBundle)
    isolation: IsolationBundle = Field(default_factory=IsolationBundle)
    extensions: EnvironmentExtensions = Field(default_factory=EnvironmentExtensions)

    @model_validator(mode="before")
    @classmethod
    def _lift_flat_or_nested(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return migrate_environment_profile_wire(lift_flat_profile_dict(data))
        return data

    # Flat wire accessors (APP-EVOL-8.2)
    @property
    def profile_id(self) -> str:
        return self.meta.profile_id

    @profile_id.setter
    def profile_id(self, value: str) -> None:
        object.__setattr__(self, "meta", self.meta.model_copy(update={"profile_id": value}))

    @property
    def spec_version(self) -> str:
        return self.meta.spec_version

    @spec_version.setter
    def spec_version(self, value: str) -> None:
        object.__setattr__(self, "meta", self.meta.model_copy(update={"spec_version": value}))

    @property
    def application_profile(self):
        return self.meta.application_profile

    @application_profile.setter
    def application_profile(self, value) -> None:
        object.__setattr__(
            self,
            "meta",
            self.meta.model_copy(update={"application_profile": value}),
        )

    @property
    def execution_mode(self) -> ExecutionMode:
        return self.meta.execution_mode

    @execution_mode.setter
    def execution_mode(self, value: ExecutionMode) -> None:
        object.__setattr__(self, "meta", self.meta.model_copy(update={"execution_mode": value}))

    @property
    def features(self):
        return self.meta.features

    @features.setter
    def features(self, value) -> None:
        object.__setattr__(self, "meta", self.meta.model_copy(update={"features": value}))

    @property
    def identity_profile(self) -> IdentityProfile:
        return self.security.identity

    @identity_profile.setter
    def identity_profile(self, value: IdentityProfile) -> None:
        object.__setattr__(
            self,
            "security",
            self.security.model_copy(update={"identity": value}),
        )

    @property
    def security_profile(self) -> ApplicationSecurityProfile:
        return self.security.application_security

    @security_profile.setter
    def security_profile(self, value: ApplicationSecurityProfile) -> None:
        object.__setattr__(
            self,
            "security",
            self.security.model_copy(update={"application_security": value}),
        )

    @property
    def guardrail_profile(self) -> GuardrailProfile:
        return self.security.guardrails

    @guardrail_profile.setter
    def guardrail_profile(self, value: GuardrailProfile) -> None:
        object.__setattr__(self, "security", self.security.model_copy(update={"guardrails": value}))

    @property
    def policy_rules(self) -> PolicyRulesProfile | None:
        return self.security.policy_rules

    @policy_rules.setter
    def policy_rules(self, value: PolicyRulesProfile | None) -> None:
        object.__setattr__(self, "security", self.security.model_copy(update={"policy_rules": value}))

    @property
    def compliance_profile(self) -> ComplianceProfile:
        return self.security.compliance

    @compliance_profile.setter
    def compliance_profile(self, value: ComplianceProfile) -> None:
        object.__setattr__(
            self,
            "security",
            self.security.model_copy(update={"compliance": value}),
        )

    @property
    def organizational_policy(self) -> OrganizationalPolicyEnvelope | None:
        return self.security.organizational_policy

    @organizational_policy.setter
    def organizational_policy(self, value: OrganizationalPolicyEnvelope | None) -> None:
        object.__setattr__(
            self,
            "security",
            self.security.model_copy(update={"organizational_policy": value}),
        )

    @property
    def integration_profile(self) -> IntegrationProfile:
        return self.capabilities.integrations

    @integration_profile.setter
    def integration_profile(self, value: IntegrationProfile) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"integrations": value}),
        )

    @property
    def tool_profile(self):
        return self.capabilities.tools

    @tool_profile.setter
    def tool_profile(self, value) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"tools": value}),
        )

    @property
    def skill_profile(self):
        return self.capabilities.skills

    @skill_profile.setter
    def skill_profile(self, value) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"skills": value}),
        )

    @property
    def llm_profile(self):
        return self.capabilities.llm

    @llm_profile.setter
    def llm_profile(self, value) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"llm": value}),
        )

    @property
    def llm_routing_profile(self) -> LLMRoutingProfile | None:
        return self.capabilities.llm_routing

    @llm_routing_profile.setter
    def llm_routing_profile(self, value: LLMRoutingProfile | None) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"llm_routing": value}),
        )

    @property
    def llm_routing_evaluating_secondary(self) -> bool:
        return self.capabilities.llm_routing_evaluating_secondary

    @llm_routing_evaluating_secondary.setter
    def llm_routing_evaluating_secondary(self, value: bool) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"llm_routing_evaluating_secondary": value}),
        )

    @property
    def modality_profile(self):
        return self.capabilities.modality

    @modality_profile.setter
    def modality_profile(self, value) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"modality": value}),
        )

    @property
    def prompt_profile(self) -> PromptProfile:
        return self.capabilities.prompt

    @prompt_profile.setter
    def prompt_profile(self, value: PromptProfile) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"prompt": value}),
        )

    @property
    def context_profile(self) -> ContextProfile:
        return self.capabilities.context

    @context_profile.setter
    def context_profile(self, value: ContextProfile) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"context": value}),
        )

    @property
    def memory_profile(self) -> MemoryProfile:
        return self.capabilities.memory

    @memory_profile.setter
    def memory_profile(self, value: MemoryProfile) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(update={"memory": value}),
        )

    @property
    def tool_selection_mode(self) -> str:
        return self.capabilities.tool_selection.mode

    @tool_selection_mode.setter
    def tool_selection_mode(self, value: str) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(
                update={
                    "tool_selection": self.capabilities.tool_selection.model_copy(
                        update={"mode": value},
                    ),
                },
            ),
        )

    @property
    def tool_selection_top_k(self) -> int:
        return self.capabilities.tool_selection.top_k

    @tool_selection_top_k.setter
    def tool_selection_top_k(self, value: int) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(
                update={
                    "tool_selection": self.capabilities.tool_selection.model_copy(
                        update={"top_k": value},
                    ),
                },
            ),
        )

    @property
    def tool_invocation_mode(self) -> str:
        return self.capabilities.tool_invocation.mode

    @tool_invocation_mode.setter
    def tool_invocation_mode(self, value: str) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(
                update={
                    "tool_invocation": self.capabilities.tool_invocation.model_copy(
                        update={"mode": value},
                    ),
                },
            ),
        )

    @property
    def max_parallel_tool_calls(self) -> int:
        return self.capabilities.tool_invocation.max_parallel

    @max_parallel_tool_calls.setter
    def max_parallel_tool_calls(self, value: int) -> None:
        object.__setattr__(
            self,
            "capabilities",
            self.capabilities.model_copy(
                update={
                    "tool_invocation": self.capabilities.tool_invocation.model_copy(
                        update={"max_parallel": value},
                    ),
                },
            ),
        )

    @property
    def reasoning_profile(self):
        return self.cognition.reasoning

    @reasoning_profile.setter
    def reasoning_profile(self, value) -> None:
        object.__setattr__(
            self,
            "cognition",
            self.cognition.model_copy(update={"reasoning": value}),
        )

    @property
    def orchestration_profile(self) -> OrchestrationProfile:
        return self.cognition.orchestration

    @orchestration_profile.setter
    def orchestration_profile(self, value: OrchestrationProfile) -> None:
        object.__setattr__(
            self,
            "cognition",
            self.cognition.model_copy(update={"orchestration": value}),
        )

    @property
    def decision_profile(self) -> DecisionProfile:
        return self.cognition.decision

    @decision_profile.setter
    def decision_profile(self, value: DecisionProfile) -> None:
        object.__setattr__(
            self,
            "cognition",
            self.cognition.model_copy(update={"decision": value}),
        )

    @property
    def adaptive_profile(self) -> AdaptiveProfile:
        return self.cognition.adaptive

    @adaptive_profile.setter
    def adaptive_profile(self, value: AdaptiveProfile) -> None:
        object.__setattr__(
            self,
            "cognition",
            self.cognition.model_copy(update={"adaptive": value}),
        )

    @property
    def evaluation_profile(self) -> EvaluationProfile:
        return self.cognition.evaluation

    @evaluation_profile.setter
    def evaluation_profile(self, value: EvaluationProfile) -> None:
        object.__setattr__(
            self,
            "cognition",
            self.cognition.model_copy(update={"evaluation": value}),
        )

    @property
    def codecraft_profile(self):
        return self.cognition.codecraft

    @codecraft_profile.setter
    def codecraft_profile(self, value) -> None:
        object.__setattr__(
            self,
            "cognition",
            self.cognition.model_copy(update={"codecraft": value}),
        )

    @property
    def reliability_profile(self) -> ReliabilityProfile:
        return self.governance.reliability

    @reliability_profile.setter
    def reliability_profile(self, value: ReliabilityProfile) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"reliability": value}),
        )

    @property
    def diagnostic_profile(self) -> DiagnosticProfile:
        return self.governance.diagnostics

    @diagnostic_profile.setter
    def diagnostic_profile(self, value: DiagnosticProfile) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"diagnostics": value}),
        )

    @property
    def observability_profile(self) -> ObservabilityProfile:
        return self.governance.observability

    @observability_profile.setter
    def observability_profile(self, value: ObservabilityProfile) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"observability": value}),
        )

    @property
    def cost_profile(self) -> CostProfile:
        return self.governance.cost

    @cost_profile.setter
    def cost_profile(self, value: CostProfile) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"cost": value}),
        )

    @property
    def scaling_profile(self) -> ScalingProfile:
        return self.governance.scaling

    @scaling_profile.setter
    def scaling_profile(self, value: ScalingProfile) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"scaling": value}),
        )

    @property
    def governance_profile(self) -> GovernanceProfile:
        return self.governance.platform

    @governance_profile.setter
    def governance_profile(self, value: GovernanceProfile) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"platform": value}),
        )

    @property
    def capability_governance_profile(self):
        return self.governance.capability

    @capability_governance_profile.setter
    def capability_governance_profile(self, value) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"capability": value}),
        )

    @property
    def agent_governance_profile(self):
        return self.governance.agent

    @agent_governance_profile.setter
    def agent_governance_profile(self, value) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"agent": value}),
        )

    @property
    def integration_governance_profile(self) -> IntegrationGovernanceProfile:
        return self.governance.integration_marketplace

    @integration_governance_profile.setter
    def integration_governance_profile(self, value: IntegrationGovernanceProfile) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"integration_marketplace": value}),
        )

    @property
    def host_deployment_profile(self) -> HostDeploymentProfile:
        return self.governance.deployment

    @host_deployment_profile.setter
    def host_deployment_profile(self, value: HostDeploymentProfile) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"deployment": value}),
        )

    @property
    def execution_boundary_export_profile(self) -> ExecutionBoundaryExportProfile | None:
        return self.governance.boundary_export

    @execution_boundary_export_profile.setter
    def execution_boundary_export_profile(
        self,
        value: ExecutionBoundaryExportProfile | None,
    ) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"boundary_export": value}),
        )

    @property
    def governance_permission_preset(self):
        return self.governance.permission_preset

    @governance_permission_preset.setter
    def governance_permission_preset(self, value) -> None:
        object.__setattr__(
            self,
            "governance",
            self.governance.model_copy(update={"permission_preset": value}),
        )

    @property
    def graph_spec(self):
        return self.topology.graph_spec

    @graph_spec.setter
    def graph_spec(self, value) -> None:
        object.__setattr__(
            self,
            "topology",
            self.topology.model_copy(update={"graph_spec": value}),
        )

    @property
    def shadow_workspace(self) -> ShadowWorkspaceProfile | None:
        return self.isolation.shadow_workspace

    @shadow_workspace.setter
    def shadow_workspace(self, value: ShadowWorkspaceProfile | None) -> None:
        object.__setattr__(
            self,
            "isolation",
            self.isolation.model_copy(update={"shadow_workspace": value}),
        )

    @property
    def sandbox(self) -> SandboxProfile | None:
        return self.isolation.sandbox

    @sandbox.setter
    def sandbox(self, value: SandboxProfile | None) -> None:
        object.__setattr__(
            self,
            "isolation",
            self.isolation.model_copy(update={"sandbox": value}),
        )

    @property
    def domain_policy_fragments(self) -> dict[str, Any]:
        return self.extensions.domain_policy_fragments.to_runtime_dict()

    @domain_policy_fragments.setter
    def domain_policy_fragments(self, value: dict[str, Any]) -> None:
        object.__setattr__(
            self,
            "extensions",
            self.extensions.model_copy(
                update={
                    "domain_policy_fragments": DomainPolicyFragments.from_runtime_dict(
                        value,
                    ),
                },
            ),
        )

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False):
        """Apply bundle-root and flat-field updates without full re-validation (APP-EVOL-8.2)."""
        if not update:
            return super().model_copy(deep=deep)
        copied = super().model_copy(deep=deep)
        for key, value in update.items():
            if key in BUNDLE_ROOT_KEYS:
                object.__setattr__(copied, key, value)
                continue
            prop = type(copied).__dict__.get(key)
            if isinstance(prop, property) and prop.fset is not None:
                prop.fset(copied, value)
            else:
                object.__setattr__(copied, key, value)
        return copied

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        nested = super().model_dump(*args, **kwargs)
        if uses_nested_profile_wire(self.meta.spec_version):
            return nested
        return flatten_profile_dict(nested)

    def bundle_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Canonical nested dump for digests and diff (APP-EVOL-8.3)."""
        return super().model_dump(mode=kwargs.get("mode", "json"))

    def with_spec_v2_wire(self) -> ApplicationEnvironmentProfile:
        """Return profile using nested canonical ``spec_version`` 2.0.0 wire (APP-EVOL-8.6)."""
        if uses_nested_profile_wire(self.meta.spec_version):
            return self
        return self.model_copy(
            update={
                "meta": self.meta.model_copy(update={"spec_version": PROFILE_SPEC_V2}),
            },
        )

    @classmethod
    def harness_memory_profile(cls) -> MemoryProfile:
        from intergrax.applications.contracts.environment_profile.presets import (
            harness_memory_profile,
        )

        return harness_memory_profile()

    def with_harness_memory(self) -> ApplicationEnvironmentProfile:
        return self.model_copy(
            update={
                "capabilities": self.capabilities.model_copy(
                    update={"memory": self.harness_memory_profile()},
                ),
            },
        )

    @classmethod
    def lab_org_virtual_workforce_defaults(
        cls,
        *,
        profile_id: str = "lab.org.virtual_workforce",
    ) -> ApplicationEnvironmentProfile:
        from intergrax.applications.contracts.org_policy import lab_strict_org_envelope

        return cls.lab_defaults(profile_id=profile_id).with_uc11_organizational_policy(
            lab_strict_org_envelope(),
        )

    def with_uc11_organizational_policy(
        self,
        envelope: OrganizationalPolicyEnvelope,
    ) -> ApplicationEnvironmentProfile:
        return self.model_copy(
            update={
                "meta": self.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
                "security": self.security.model_copy(update={"organizational_policy": envelope}),
            },
        )

    @classmethod
    def lab_defaults(
        cls,
        *,
        profile_id: str = "lab.default",
        harness_tools: bool = True,
    ) -> ApplicationEnvironmentProfile:
        from intergrax.applications.contracts.environment_profile.presets import (
            build_lab_defaults,
        )

        return cls(**build_lab_defaults(profile_id=profile_id, harness_tools=harness_tools))

    @classmethod
    def harness_production_defaults(
        cls,
        *,
        profile_id: str = "harness.production",
        harness_tools: bool = True,
        secrets_slug: str = "doppler",
        enable_grafana_stack: bool = True,
    ) -> ApplicationEnvironmentProfile:
        from intergrax.applications.contracts.environment_profile.presets import (
            build_harness_production_defaults,
        )

        return cls(
            **build_harness_production_defaults(
                profile_id=profile_id,
                harness_tools=harness_tools,
                secrets_slug=secrets_slug,
                enable_grafana_stack=enable_grafana_stack,
            ),
        )

    @classmethod
    def strict_multi_agent_defaults(
        cls,
        *,
        profile_id: str = "strict.multi_agent",
    ) -> ApplicationEnvironmentProfile:
        base = cls.lab_defaults(profile_id=profile_id)
        regulated = CognitionBundle.regulated()
        return base.model_copy(
            update={
                "meta": base.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
                "cognition": base.cognition.model_copy(
                    update={
                        "orchestration": OrchestrationProfile(
                            merge_strategy="structured_json",
                            max_parallel_nodes=8,
                            max_inflight_nodes=8,
                            max_run_retries=1,
                        ),
                        "decision": regulated.decision,
                        "evaluation": regulated.evaluation,
                    },
                ),
            },
        )

    def with_reference_host_platform_defaults(
        self,
        *,
        multi_agent_critic: bool = False,
    ) -> ApplicationEnvironmentProfile:
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
            "cognition": self.cognition.model_copy(
                update={"orchestration": orchestration},
            ),
            "governance": self.governance.model_copy(
                update={"reliability": reliability},
            ),
        }
        if multi_agent_critic:
            strict = type(self).strict_multi_agent_defaults(profile_id=self.profile_id)
            updates["meta"] = self.meta.model_copy(
                update={"execution_mode": strict.execution_mode},
            )
            updates["cognition"] = self.cognition.model_copy(
                update={
                    "decision": strict.decision_profile,
                    "evaluation": strict.evaluation_profile,
                    "orchestration": orchestration.model_copy(
                        update={
                            "merge_strategy": strict.orchestration_profile.merge_strategy,
                            "max_run_retries": strict.orchestration_profile.max_run_retries,
                            "max_parallel_nodes": strict.orchestration_profile.max_parallel_nodes,
                            "max_inflight_nodes": strict.orchestration_profile.max_inflight_nodes,
                        },
                    ),
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
        base = cls.lab_defaults(profile_id=profile_id)
        return base.model_copy(
            update={
                "cognition": base.cognition.model_copy(
                    update={
                        "orchestration": OrchestrationProfile(
                            long_running_enabled=True,
                            merge_strategy="structured_json",
                            max_parallel_nodes=max_parallel_nodes,
                            max_inflight_nodes=max_parallel_nodes,
                        ),
                    },
                ),
                "governance": base.governance.model_copy(
                    update={
                        "reliability": ReliabilityProfile(
                            long_running_scheduler_enabled=True,
                            checkpoint_interval_steps=1,
                        ),
                    },
                ),
            },
        )

    @classmethod
    def swarm_exploration_defaults(
        cls,
        *,
        profile_id: str = "swarm.exploration",
        max_parallel_nodes: int = 16,
    ) -> ApplicationEnvironmentProfile:
        from intergrax.applications.contracts.environment_profile.presets import (
            swarm_coordination_pattern_value,
        )

        base = cls.lab_defaults(profile_id=profile_id)
        return base.model_copy(
            update={
                "cognition": base.cognition.model_copy(
                    update={
                        "orchestration": OrchestrationProfile(
                            merge_strategy="structured_json",
                            max_parallel_nodes=max_parallel_nodes,
                            max_inflight_nodes=max_parallel_nodes,
                            coordination_pattern=swarm_coordination_pattern_value(),
                        ),
                    },
                ),
            },
        )

    @classmethod
    def _product_integration_profile(cls) -> IntegrationProfile:
        from intergrax.applications.contracts.environment_profile.presets import (
            product_integration_profile,
        )

        return product_integration_profile()

    @classmethod
    def _product_modality_profile(cls):
        from intergrax.contracts.modality_profile import production_plane_c_modality_profile

        return production_plane_c_modality_profile()

    @classmethod
    def _product_budget_reaction(cls) -> BudgetReactionProfile:
        from intergrax.applications.contracts.environment_profile.presets import (
            product_budget_reaction,
        )

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
        from intergrax.applications.contracts.environment_profile.presets import (
            build_product_defaults,
        )

        return cls(
            **build_product_defaults(
                profile_id=profile_id,
                skill_bundles=skill_bundles,
                tool_ids=tool_ids,
                domain_fragments=domain_fragments,
            ),
        )


ApplicationEnvironmentProfile.model_rebuild()
SecurityEnvelope.model_rebuild()
