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
from intergrax.tools.providers.catalog.bundle import CATALOG_BUNDLE_ID
from intergrax.tools.providers.confluence.bundle import CONFLUENCE_BUNDLE_ID
from intergrax.tools.providers.context_tool.bundle import CONTEXT_BUNDLE_ID
from intergrax.tools.providers.document.bundle import DOCUMENT_BUNDLE_ID
from intergrax.tools.providers.harness.bundle import HARNESS_BUNDLE_ID
from intergrax.tools.providers.health.bundle import HEALTH_BUNDLE_ID
from intergrax.tools.providers.knowledge.bundle import KNOWLEDGE_BUNDLE_ID
from intergrax.tools.providers.ltm.bundle import LTM_BUNDLE_ID
from intergrax.tools.providers.memory.bundle import MEMORY_BUNDLE_ID
from intergrax.tools.providers.ml.bundle import ML_BUNDLE_ID
from intergrax.tools.providers.observability.bundle import OBSERVABILITY_BUNDLE_ID
from intergrax.tools.providers.openai_vector_store.bundle import OPENAI_VECTOR_STORE_BUNDLE_ID
from intergrax.tools.providers.rag.bundle import RAG_BUNDLE_ID
from intergrax.tools.providers.sandbox.bundle import SANDBOX_BUNDLE_ID
from intergrax.tools.providers.security.bundle import SECURITY_BUNDLE_ID
from intergrax.tools.providers.skill_tool.bundle import SKILL_BUNDLE_ID
from intergrax.tools.providers.speech.bundle import SPEECH_BUNDLE_ID
from intergrax.tools.providers.storage.bundle import STORAGE_BUNDLE_ID
from intergrax.tools.providers.vision.bundle import VISION_BUNDLE_ID
from intergrax.tools.providers.websearch.bundle import WEBSEARCH_BUNDLE_ID
from intergrax.tools.providers.workflow.bundle import WORKFLOW_BUNDLE_ID
from intergrax.tools.providers.workspace.bundle import WORKSPACE_BUNDLE_ID
from intergrax.tools.registry.profile import ToolProfile

_LAB_REFERENCE_TOOL_BUNDLE_IDS: tuple[str, ...] = (
    CATALOG_BUNDLE_ID,
    CONFLUENCE_BUNDLE_ID,
    CONTEXT_BUNDLE_ID,
    DOCUMENT_BUNDLE_ID,
    HARNESS_BUNDLE_ID,
    HEALTH_BUNDLE_ID,
    KNOWLEDGE_BUNDLE_ID,
    LTM_BUNDLE_ID,
    MEMORY_BUNDLE_ID,
    ML_BUNDLE_ID,
    OBSERVABILITY_BUNDLE_ID,
    OPENAI_VECTOR_STORE_BUNDLE_ID,
    RAG_BUNDLE_ID,
    SECURITY_BUNDLE_ID,
    SKILL_BUNDLE_ID,
    STORAGE_BUNDLE_ID,
    VISION_BUNDLE_ID,
    WEBSEARCH_BUNDLE_ID,
    WORKFLOW_BUNDLE_ID,
    WORKSPACE_BUNDLE_ID,
)

_HARNESS_OPTIONAL_TOOL_BUNDLE_IDS: tuple[str, ...] = (
    SANDBOX_BUNDLE_ID,
    SPEECH_BUNDLE_ID,
)


def harness_memory_profile() -> MemoryProfile:
    return MemoryProfile(
        enable_user_memory=True,
        enable_org_memory=True,
        enable_long_term_memory=True,
        enable_task_memory=True,
    )


def lab_reference_tool_profile(*, harness_tools: bool = True) -> ToolProfile:
    """Explicit reference-lab host tool availability (least privilege, no catalog-wide grant)."""
    enabled_bundles = list(_LAB_REFERENCE_TOOL_BUNDLE_IDS)
    if harness_tools:
        enabled_bundles.extend(_HARNESS_OPTIONAL_TOOL_BUNDLE_IDS)
    return ToolProfile(enabled_bundles=enabled_bundles)


def harness_platform_tool_profile() -> ToolProfile:
    """Tool availability for harness-only platform skill hosts (pairs with harness skill bundle)."""
    return lab_reference_tool_profile(harness_tools=False)


def harness_lab_capability_bundle(*, harness_tools: bool = True) -> CapabilityBundle:
    """Shared lab harness tools/skills/integration stack (legal, research, lab hosts)."""
    from intergrax.applications._shared.skill_wiring import lab_skill_profile

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
