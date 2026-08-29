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
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.routing import LLMRoutingProfile
from intergrax.runtime.adaptive.contracts import UtilityWeights
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
from intergrax.runtime.capacity.contracts import ScalingPolicy

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
    uses_nested_profile_wire,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    AdaptiveProfile,
    ApplicationSecurityProfile,
    ComplianceProfile,
    ContextProfile,
    CostProfile,
    CriticProfile,
    CriticVerificationScopes,
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
            return lift_flat_profile_dict(data)
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
    def critic_profile(self) -> CriticProfile:
        return self.cognition.critic

    @critic_profile.setter
    def critic_profile(self, value: CriticProfile) -> None:
        object.__setattr__(
            self,
            "cognition",
            self.cognition.model_copy(update={"critic": value}),
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
        from intergrax.applications._shared.reference_capability_bundle import (
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
        from intergrax.applications._shared.reference_capability_bundle import (
            harness_lab_capability_bundle,
        )

        return cls(
            meta=HostMeta.lab(profile_id=profile_id),
            security=SecurityEnvelope.lab(),
            capabilities=harness_lab_capability_bundle(harness_tools=harness_tools),
            cognition=CognitionBundle.lab(),
            governance=GovernanceBundle.lab(),
            isolation=IsolationBundle.lab(),
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
        from intergrax.applications._shared.skill_wiring import lab_skill_profile
        from intergrax.integrations.registry import presets

        base = cls.lab_defaults(profile_id=profile_id, harness_tools=harness_tools)
        return base.model_copy(
            update={
                "meta": base.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
                "capabilities": base.capabilities.model_copy(
                    update={
                        "integrations": presets.harness_production_stack(
                            secrets_slug=secrets_slug,
                            enable_grafana_stack=enable_grafana_stack,
                        ),
                        "skills": lab_skill_profile(),
                    },
                ),
                "governance": base.governance.model_copy(
                    update={
                        "observability": ObservabilityProfile(
                            trace_sqlite_enabled=True,
                            otel_enabled=True,
                            metrics_plugins_enabled=True,
                            debug_surface_override=False,
                        ),
                    },
                ),
                "cognition": base.cognition.model_copy(
                    update={
                        "adaptive": AdaptiveProfile(
                            enabled=False,
                            mode="observe",
                            feature_flag_slug="unleash",
                            rollout_flag_key="harness.adaptive.recommend",
                        ),
                    },
                ),
                "security": base.security.model_copy(
                    update={"identity": IdentityProfile(require_api_key=True)},
                ),
            },
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
                            max_run_retries=1,
                        ),
                        "critic": regulated.critic,
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
                    "critic": strict.critic_profile.model_copy(
                        update={
                            "require_critic_on_completion": True,
                            "semantic_judge_enabled": bool(
                                strict.critic_profile.default_rubric_ref,
                            ),
                        },
                    ),
                    "evaluation": strict.evaluation_profile,
                    "orchestration": orchestration.model_copy(
                        update={
                            "merge_strategy": strict.orchestration_profile.merge_strategy,
                            "max_run_retries": strict.orchestration_profile.max_run_retries,
                            "max_parallel_nodes": strict.orchestration_profile.max_parallel_nodes,
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
        from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern

        base = cls.lab_defaults(profile_id=profile_id)
        return base.model_copy(
            update={
                "cognition": base.cognition.model_copy(
                    update={
                        "orchestration": OrchestrationProfile(
                            merge_strategy="structured_json",
                            max_parallel_nodes=max_parallel_nodes,
                            max_inflight_nodes=max_parallel_nodes,
                            coordination_pattern=CoordinationPattern.SWARM.value,
                        ),
                    },
                ),
            },
        )

    @classmethod
    def _product_integration_profile(cls) -> IntegrationProfile:
        from intergrax.integrations.registry.presets import POSTGRESQL

        return IntegrationProfile(
            relational_store=POSTGRESQL,
            options={"postgresql": {"tenant_schema": "tenant_default"}},
        )

    @classmethod
    def _product_modality_profile(cls):
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
        from intergrax.skills.registry.profile import SkillProfile
        from intergrax.tools.registry.profile import ToolProfile

        bundles = skill_bundles or []
        tools = tool_ids or []
        return cls(
            meta=HostMeta.product(profile_id=profile_id),
            security=SecurityEnvelope.strict(),
            capabilities=CapabilityBundle(
                integrations=cls._product_integration_profile(),
                tools=ToolProfile(enabled=tools) if tools else ToolProfile(),
                skills=SkillProfile(enabled_bundles=bundles) if bundles else SkillProfile(),
                modality=cls._product_modality_profile(),
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
            cognition=CognitionBundle(
                orchestration=OrchestrationProfile(long_running_enabled=True),
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
            governance=GovernanceBundle(
                reliability=ReliabilityProfile(
                    long_running_scheduler_enabled=True,
                    compensation_enabled=True,
                    partial_results_enabled=True,
                    middleware_hook_timeout_seconds=0.25,
                    recovery_contract=standard_strict_product_recovery_contract(),
                ),
                observability=GovernanceBundle.production_slo().observability,
                cost=CostProfile(
                    max_total_tokens=32_000,
                    max_llm_calls=32,
                    max_tool_calls=64,
                    forecasting_enabled=True,
                    optimization_recommendations_enabled=True,
                    tenant_fairness_quotas_enabled=True,
                    budget_reaction=cls._product_budget_reaction(),
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
            isolation=IsolationBundle.product(),
            extensions=EnvironmentExtensions(
                domain_policy_fragments=DomainPolicyFragments.from_runtime_dict(
                    domain_fragments,
                ),
            ),
        )


ApplicationEnvironmentProfile.model_rebuild()
SecurityEnvelope.model_rebuild()
