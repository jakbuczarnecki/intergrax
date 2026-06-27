# © Artur Czarnecki. All rights reserved.

"""Map Tier-3 application contracts to neutral runtime/agent boundaries."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_run_binding import AgentRunBinding
from intergrax.contracts.host_profile_slices import (
    ContextProfile,
    CostProfile,
    ExecutionBoundaryExportProfile,
    MemoryProfile,
    ReliabilityProfile,
)
from intergrax.contracts.org_policy import OrganizationalPolicyEnvelope
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile


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
        export_profile = ExecutionBoundaryExportProfile.model_validate(
            profile.execution_boundary_export_profile.model_dump(mode="json"),
        )
    return RuntimeEnvironmentProfile(
        profile_id=profile.profile_id,
        execution_mode=profile.execution_mode,
        memory_profile=MemoryProfile.model_validate(profile.memory_profile.model_dump(mode="json")),
        context_profile=ContextProfile.model_validate(profile.context_profile.model_dump(mode="json")),
        cost_profile=CostProfile.model_validate(profile.cost_profile.model_dump(mode="json")),
        llm_profile=profile.llm_profile,
        llm_routing_profile=profile.llm_routing_profile,
        reliability_profile=ReliabilityProfile.model_validate(
            profile.reliability_profile.model_dump(mode="json"),
        ),
        execution_boundary_export_profile=export_profile,
        organizational_policy=org_policy,
    )
