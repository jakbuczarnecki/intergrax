# © Artur Czarnecki. All rights reserved.

"""IDEAL §17 umbrella — typed Tier-3 application environment (Phase H-APP.1.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.applications.contracts.application_host import ApplicationFeatures, ApplicationProfile
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.modality.modality_profile import ModalityProfile, lab_default_modality_profile
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.skills.registry.profile import SkillProfile
from intergrax.tools.registry.profile import ToolProfile


class IdentityProfile(BaseModel):
    """Harness identity posture for a Tier-3 host (Phase H-APP.2.1)."""

    model_config = ConfigDict(extra="forbid")

    require_api_key: bool = False
    api_key_env: str = "INTERGRAX_HARNESS_API_KEY"
    tenant_required: bool = False
    role_claims_header: str | None = None
    service_identities: dict[str, str] = Field(default_factory=dict)


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


class ContextProfile(BaseModel):
    """Context assembly defaults for Nexus (Phase H-APP.4.1)."""

    model_config = ConfigDict(extra="forbid")

    assembly_options: TaskContextAssemblyOptions = Field(
        default_factory=TaskContextAssemblyOptions
    )
    budget_policy: ContextBudgetPolicy | None = None
    enable_rag: bool = True
    enable_websearch: bool = True


class MemoryProfile(BaseModel):
    """Task/org memory flags (Phase H-APP.4.2)."""

    model_config = ConfigDict(extra="forbid")

    enable_user_memory: bool = False
    enable_org_memory: bool = False
    enable_long_term_memory: bool = False
    retention_days: int | None = Field(default=None, ge=1)
    scope_boundary: str = "tenant"


class ReliabilityProfile(BaseModel):
    """Idempotency, circuit breaker, checkpoint, scheduler (Phase H-APP.4.5)."""

    model_config = ConfigDict(extra="forbid")

    idempotency_enabled: bool = True
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    checkpoint_interval_steps: int = Field(default=1, ge=1)
    long_running_scheduler_enabled: bool = False


class ObservabilityProfile(BaseModel):
    """Trace, OTEL, metrics, optional product debug (Phase H-APP.4.8)."""

    model_config = ConfigDict(extra="forbid")

    trace_sqlite_enabled: bool = True
    otel_enabled: bool = False
    metrics_plugins_enabled: bool = True
    debug_surface_override: bool | None = None


class OrchestrationProfile(BaseModel):
    """Nexus loop composition overrides (Phase H-APP.3.1)."""

    model_config = ConfigDict(extra="forbid")

    planner_kind: str | None = None
    classifier_kind: str | None = None
    retry_policy_name: str | None = None
    long_running_enabled: bool = False
    max_delegation_depth: int = Field(default=4, ge=1, le=32)


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
    context_profile: ContextProfile = Field(default_factory=ContextProfile)
    memory_profile: MemoryProfile = Field(default_factory=MemoryProfile)
    reliability_profile: ReliabilityProfile = Field(default_factory=ReliabilityProfile)
    observability_profile: ObservabilityProfile = Field(default_factory=ObservabilityProfile)
    orchestration_profile: OrchestrationProfile = Field(default_factory=OrchestrationProfile)
    identity_profile: IdentityProfile = Field(default_factory=IdentityProfile)
    security_profile: ApplicationSecurityProfile = Field(
        default_factory=ApplicationSecurityProfile
    )
    policy_rules: PolicyRulesProfile | None = None
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    graph_spec: ApplicationGraphSpec | None = None
    shadow_workspace: ShadowWorkspaceProfile | None = None
    sandbox: SandboxProfile | None = None
    features: ApplicationFeatures = Field(default_factory=ApplicationFeatures.lab_defaults)
    domain_policy_fragments: dict[str, Any] = Field(default_factory=dict)

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
            modality_profile=lab_default_modality_profile(),
            llm_profile=LLMProfile.lab(),
            context_profile=ContextProfile(enable_rag=True, enable_websearch=True),
            reliability_profile=ReliabilityProfile(
                long_running_scheduler_enabled=True,
                idempotency_enabled=True,
            ),
            observability_profile=ObservabilityProfile(
                trace_sqlite_enabled=True,
                otel_enabled=False,
                metrics_plugins_enabled=True,
                debug_surface_override=True,
            ),
            orchestration_profile=OrchestrationProfile(long_running_enabled=True),
            identity_profile=IdentityProfile(require_api_key=False),
            shadow_workspace=ShadowWorkspaceProfile(),
            sandbox=SandboxProfile(enable_exec_tool=True),
            features=ApplicationFeatures.lab_defaults(),
            execution_mode=ExecutionMode.BALANCED,
        )

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
            integration_profile=IntegrationProfile(),
            tool_profile=ToolProfile(enabled=tools) if tools else ToolProfile(),
            skill_profile=SkillProfile(enabled_bundles=bundles) if bundles else SkillProfile(),
            llm_profile=None,
            context_profile=ContextProfile(enable_rag=False, enable_websearch=False),
            reliability_profile=ReliabilityProfile(long_running_scheduler_enabled=False),
            observability_profile=ObservabilityProfile(
                trace_sqlite_enabled=True,
                debug_surface_override=False,
            ),
            features=ApplicationFeatures.product_defaults(),
            execution_mode=ExecutionMode.STRICT,
            domain_policy_fragments=dict(domain_fragments or {}),
        )
