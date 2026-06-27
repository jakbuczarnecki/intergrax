# © Artur Czarnecki. All rights reserved.

"""Neutral runtime environment profile for agent/runtime merge."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.contracts.host_profile_slices import (
    ContextProfile,
    CostProfile,
    ExecutionBoundaryExportProfile,
    MemoryProfile,
    ReliabilityProfile,
)
from intergrax.contracts.org_policy import OrganizationalPolicyEnvelope
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import LLMRoutingProfile


class RuntimeEnvironmentProfile(BaseModel):
    """Minimal environment slice passed from application hosts into agent/runtime."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = "runtime.default"
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    memory_profile: MemoryProfile = Field(default_factory=MemoryProfile)
    context_profile: ContextProfile = Field(default_factory=ContextProfile)
    cost_profile: CostProfile = Field(default_factory=CostProfile)
    llm_profile: LLMProfile | None = None
    llm_routing_profile: LLMRoutingProfile | None = None
    reliability_profile: ReliabilityProfile = Field(default_factory=ReliabilityProfile)
    execution_boundary_export_profile: ExecutionBoundaryExportProfile | None = None
    organizational_policy: OrganizationalPolicyEnvelope | None = None
