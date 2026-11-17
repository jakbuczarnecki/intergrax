# © Artur Czarnecki. All rights reserved.

"""Map Tier-3 application contracts to neutral runtime/agent boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    AdaptiveProfile as Tier3AdaptiveProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ApplicationSecurityProfile as Tier3ApplicationSecurityProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ContextDecisionProfile as Tier3ContextDecisionProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ContextProfile as Tier3ContextProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    CostProfile as Tier3CostProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    EvaluationProfile as Tier3EvaluationProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ExecutionBoundaryExportProfile as Tier3ExecutionBoundaryExportProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    GuardrailProfile as Tier3GuardrailProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    MemoryProfile as Tier3MemoryProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    OrchestrationProfile as Tier3OrchestrationProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    PolicyRulesProfile as Tier3PolicyRulesProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ReliabilityProfile as Tier3ReliabilityProfile,
)
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_run_binding import AgentRunBinding
from intergrax.contracts.host_profile_slices import (
    AdaptiveProfile,
    ApplicationSecurityProfile,
    ContextDecisionProfile,
    ContextProfile,
    CostProfile,
    EvaluationProfile,
    ExecutionBoundaryExportProfile,
    GuardrailProfile,
    MemoryProfile,
    OrchestrationProfile,
    PolicyRulesProfile,
    ReliabilityProfile,
)
from intergrax.contracts.org_policy import OrganizationalPolicyEnvelope
from intergrax.contracts.profile_projection import ProfileProjection
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile


def _shared_module(name: str) -> str:
    """Reason text: the field is read from the full Tier-3 profile by ``intergrax/applications/_shared/<name>.py``."""
    return f"consumed from the full Tier-3 profile by intergrax/applications/_shared/{name}.py"


# Tier-3-only fields of each mirrored pair. Every field is read from the full Tier-3
# profile by the named module; none exists on the neutral Tier-0 slice (X-A8-003).
_MEMORYPROFILE_NOT_PROJECTED: Mapping[str, str] = MappingProxyType(
    {
        "consolidation_mode": _shared_module("memory_runtime_bridge"),
        "enable_entity_graph_memory": _shared_module("entity_graph_wiring"),
        "enable_long_horizon_memory": _shared_module("long_horizon_memory_wiring"),
        "enable_procedural_memory": _shared_module("memory_wiring"),
        "enable_session_vector_index": _shared_module("memory_provider_admission"),
        "entity_temporal_memory_store_plugin_id": _shared_module("entity_graph_wiring"),
        "include_cross_session_episodic": _shared_module("memory_runtime_bridge"),
        "long_horizon_memory_store_plugin_id": _shared_module("long_horizon_memory_wiring"),
        "procedural_memory_store_plugin_id": _shared_module("procedural_memory_wiring"),
        "retention_days": _shared_module("memory_runtime_bridge"),
        "scope_boundary": _shared_module("memory_runtime_bridge"),
        "session_index_roles": _shared_module("memory_vector_wiring"),
        "session_index_score_threshold": _shared_module("memory_runtime_bridge"),
        "session_index_top_k": _shared_module("memory_runtime_bridge"),
        "session_storage_plugin_id": _shared_module("memory_wiring"),
        "user_profile_store_plugin_id": _shared_module("memory_wiring"),
        "vector_index_namespace": _shared_module("memory_runtime_bridge"),
    },
)

_COSTPROFILE_NOT_PROJECTED: Mapping[str, str] = MappingProxyType(
    {
        "budget_enforcement_enabled": _shared_module("budget_wiring"),
        "enforcement_mode": _shared_module("cost_runtime_bridge"),
        "forecasting_enabled": _shared_module("cost_forecast_wiring"),
        "max_llm_calls": _shared_module("cost_assembly_resolver"),
        "max_planner_iterations": _shared_module("cost_assembly_resolver"),
        "max_tool_calls": _shared_module("cost_assembly_resolver"),
        "optimization_recommendations_enabled": _shared_module("cost_optimization_wiring"),
        "quota_degrade_threshold_ratio": _shared_module("cost_assembly_resolver"),
        "tenant_fairness_quotas_enabled": _shared_module("tenant_quota_wiring"),
    },
)

_RELIABILITYPROFILE_NOT_PROJECTED: Mapping[str, str] = MappingProxyType(
    {
        "attempt_lifecycle_persistence_provider": _shared_module("host_orchestration_environment_resolution"),
        "compensation_enabled": _shared_module("compensation_wiring"),
        "delegated_invocation_correlation_durability": _shared_module("delegated_invocation_correlation_wiring"),
        "dependency_concurrency_admission": _shared_module("declarative_tool_wiring"),
        "execution_lineage_persistence_provider": _shared_module("diagnostic_read_wiring"),
        "execution_terminal_persistence_provider": _shared_module("host_orchestration_environment_resolution"),
        "middleware_hook_timeout_seconds": _shared_module("application_host_wiring"),
        "partial_results_enabled": _shared_module("partial_results_wiring"),
        "recovery_contract": _shared_module("recovery_contract_wiring"),
    },
)

_POLICYRULESPROFILE_NOT_PROJECTED: Mapping[str, str] = MappingProxyType(
    {
        "allowed_handler_ids": _shared_module("policy_wiring"),
        "policy_enforcement_mode": _shared_module("policy_wiring"),
    },
)

_ADAPTIVEPROFILE_NOT_PROJECTED: Mapping[str, str] = MappingProxyType(
    {
        "business_outcome_webhook": _shared_module("business_outcome_webhook_verifier"),
    },
)


def _declare[TargetT: BaseModel](
    target: type[TargetT],
    *,
    not_projected: Mapping[str, str] | None = None,
) -> ProfileProjection[TargetT]:
    """Declare one pair: every Tier-0 field is projected; Tier-3-only fields carry a reason."""
    return ProfileProjection(
        name=target.__name__,
        target=target,
        projected=tuple(target.model_fields),
        not_projected=not_projected if not_projected is not None else MappingProxyType({}),
    )


POLICY_RULES_PROJECTION = _declare(PolicyRulesProfile, not_projected=_POLICYRULESPROFILE_NOT_PROJECTED)
GUARDRAIL_PROJECTION = _declare(GuardrailProfile)
CONTEXT_DECISION_PROJECTION = _declare(ContextDecisionProfile)
CONTEXT_PROJECTION = _declare(ContextProfile)
MEMORY_PROJECTION = _declare(MemoryProfile, not_projected=_MEMORYPROFILE_NOT_PROJECTED)
COST_PROJECTION = _declare(CostProfile, not_projected=_COSTPROFILE_NOT_PROJECTED)
RELIABILITY_PROJECTION = _declare(ReliabilityProfile, not_projected=_RELIABILITYPROFILE_NOT_PROJECTED)
EXECUTION_BOUNDARY_EXPORT_PROJECTION = _declare(ExecutionBoundaryExportProfile)
APPLICATION_SECURITY_PROJECTION = _declare(ApplicationSecurityProfile)
EVALUATION_PROJECTION = _declare(EvaluationProfile)
ADAPTIVE_PROJECTION = _declare(AdaptiveProfile, not_projected=_ADAPTIVEPROFILE_NOT_PROJECTED)
ORCHESTRATION_PROJECTION = _declare(OrchestrationProfile)

# The twelve mirrored Tier-0 / Tier-3 pairs (decision DP-17): (declared projection, Tier-3 source model).
PROFILE_PROJECTIONS: tuple[tuple[ProfileProjection[BaseModel], type[BaseModel]], ...] = (
    (POLICY_RULES_PROJECTION, Tier3PolicyRulesProfile),
    (GUARDRAIL_PROJECTION, Tier3GuardrailProfile),
    (CONTEXT_DECISION_PROJECTION, Tier3ContextDecisionProfile),
    (CONTEXT_PROJECTION, Tier3ContextProfile),
    (MEMORY_PROJECTION, Tier3MemoryProfile),
    (COST_PROJECTION, Tier3CostProfile),
    (RELIABILITY_PROJECTION, Tier3ReliabilityProfile),
    (EXECUTION_BOUNDARY_EXPORT_PROJECTION, Tier3ExecutionBoundaryExportProfile),
    (APPLICATION_SECURITY_PROJECTION, Tier3ApplicationSecurityProfile),
    (EVALUATION_PROJECTION, Tier3EvaluationProfile),
    (ADAPTIVE_PROJECTION, Tier3AdaptiveProfile),
    (ORCHESTRATION_PROJECTION, Tier3OrchestrationProfile),
)

for _projection_entry, _source_class in PROFILE_PROJECTIONS:
    # Fail closed at import: a Tier-3 field without a map entry breaks every import of this module.
    _projection_entry.verify(_source_class)


def agent_binding_to_run_binding(binding: AgentBinding | None) -> AgentRunBinding | None:
    """Convert application roster binding to neutral agent runtime slice."""
    if binding is None:
        return None
    return AgentRunBinding(
        memory_scope_override=binding.memory_scope_override,
        rag_collection_override=binding.rag_collection_override,
        tool_allowlist_extra=list(binding.tool_allowlist_extra),
        tool_denylist=list(binding.tool_denylist),
        org_role_id=binding.org_role_id,
        budget_slice=binding.budget_slice,
    )


def application_profile_to_runtime_profile(
    profile: ApplicationEnvironmentProfile | None,
) -> RuntimeEnvironmentProfile | None:
    """Convert full application environment profile to runtime merge slice."""
    if profile is None:
        return None
    org_policy: OrganizationalPolicyEnvelope | None = None
    if profile.organizational_policy is not None:
        org_policy = OrganizationalPolicyEnvelope.model_validate(
            profile.organizational_policy.model_dump(mode="json"),
        )
    export_profile: ExecutionBoundaryExportProfile | None = None
    if profile.execution_boundary_export_profile is not None:
        export_profile = EXECUTION_BOUNDARY_EXPORT_PROJECTION.project(
            profile.execution_boundary_export_profile,
        )
    return RuntimeEnvironmentProfile(
        profile_id=profile.profile_id,
        execution_mode=profile.execution_mode,
        memory_profile=MEMORY_PROJECTION.project(profile.memory_profile),
        context_profile=CONTEXT_PROJECTION.project(profile.context_profile),
        cost_profile=COST_PROJECTION.project(profile.cost_profile),
        llm_profile=profile.llm_profile,
        llm_routing_profile=profile.llm_routing_profile,
        reliability_profile=RELIABILITY_PROJECTION.project(profile.reliability_profile),
        execution_boundary_export_profile=export_profile,
        organizational_policy=org_policy,
    )
