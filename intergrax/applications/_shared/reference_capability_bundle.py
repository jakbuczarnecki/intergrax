# © Artur Czarnecki. All rights reserved.

"""Reusable CapabilityBundle presets for Tier-3 reference hosts (APP-EVOL-8.5)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile.bundles import CapabilityBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    ContextProfile,
    MemoryProfile,
    ToolSelectionConfig,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.modality.modality_profile import lab_default_modality_profile
from intergrax.tools.registry.profile import ToolProfile


def harness_memory_profile() -> MemoryProfile:
    return MemoryProfile(
        enable_user_memory=True,
        enable_org_memory=True,
        enable_long_term_memory=True,
        enable_task_memory=True,
    )


def harness_lab_capability_bundle(*, harness_tools: bool = True) -> CapabilityBundle:
    """Shared lab harness tools/skills/integration stack (legal, research, lab hosts)."""
    from intergrax.applications._shared.skill_wiring import lab_skill_profile

    del harness_tools  # integrations/features use this flag; skills need full catalog tools.
    return CapabilityBundle(
        integrations=IntegrationProfile.lab_harness_preset(),
        tools=ToolProfile(register_all_catalog_bundles=True),
        skills=lab_skill_profile(),
        llm=LLMProfile.lab(),
        modality=lab_default_modality_profile(),
        context=ContextProfile(enable_rag=True, enable_websearch=True),
        memory=harness_memory_profile(),
        tool_selection=ToolSelectionConfig(mode="skill_pack"),
    )
