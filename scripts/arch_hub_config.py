# © Artur Czarnecki. All rights reserved.
"""Per-domain architecture hub split configuration (F4 / F4-C)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchSplitConfig:
    domain: str
    hub_section_max: int = 0
    production_section_min: int = 40
    extra_hub_sections: tuple[int, ...] = ()
    h2_satellite_markers: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    numbered_h2_max: int | None = None
    subsection_major: int | None = None
    subsection_minor_max: int | None = None


def _num(domain: str, numbered_h2_max: int, **kwargs: object) -> ArchSplitConfig:
    return ArchSplitConfig(domain=domain, numbered_h2_max=numbered_h2_max, **kwargs)  # type: ignore[arg-type]


ACP = ArchSplitConfig(
    domain="AGENT_CONTRACTS_AND_ASSEMBLY",
    hub_section_max=21,
)

TIER3 = ArchSplitConfig(
    domain="TIER3_APPLICATION_ENVIRONMENT",
    hub_section_max=18,
    extra_hub_sections=(45,),
)

PLATFORM_ARCH = ArchSplitConfig(
    domain="PLATFORM_FOUNDATION",
    hub_section_max=6,
    production_section_min=43,
)

TOOLS_ARCH = ArchSplitConfig(
    domain="TOOLS",
    h2_satellite_markers=(
        ("## Tool selection — strategies and layers", "selection_and_plugins"),
        ("## Runtime configuration reference", "runtime_config_reference"),
    ),
)

UAEP_ARCH = ArchSplitConfig(
    domain="UNIFIED_EXECUTION_RUNTIME",
    subsection_major=42,
    subsection_minor_max=7,
)

NEXUS_ARCH = ArchSplitConfig(
    domain="NEXUS_EXECUTION_FLOW",
    numbered_h2_max=8,
)

ORCH_ARCH = ArchSplitConfig(
    domain="ORCHESTRATION",
    hub_section_max=26,
)

INTEGRATIONS_ARCH = ArchSplitConfig(
    domain="INTEGRATIONS",
    h2_satellite_markers=(("## Catalog", "provider_catalog"),),
)

# F4-C wave 2 — numbered ## N. hubs (extended § → arch/*_extended_depth.md)
ADAPTIVE_HARNESS_INTELLIGENCE_ARCH = _num("ADAPTIVE_HARNESS_INTELLIGENCE", 7)
CODE_CRAFT_ARCH = _num("CODE_CRAFT", 6)
CONTEXT_ENGINEERING_ARCH = _num("CONTEXT_ENGINEERING", 7)
CRITIC_VERIFICATION_ARCH = _num("CRITIC_VERIFICATION", 6)
ELASTIC_CAPACITY_AND_SCALING_ARCH = _num("ELASTIC_CAPACITY_AND_SCALING", 7)
MEMORY_ARCH = _num("MEMORY", 7)
OBSERVABILITY_ARCH = _num("OBSERVABILITY", 4)
REASONING_AND_COGNITION_ARCH = _num("REASONING_AND_COGNITION", 7)

RELIABILITY_FAILURE_AND_HITL_ARCH = ArchSplitConfig(
    domain="RELIABILITY_FAILURE_AND_HITL",
    hub_section_max=32,
    production_section_min=35,
)

EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_ARCH = ArchSplitConfig(
    domain="EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE",
    hub_section_max=41,
    production_section_min=43,
)

LLM_ADAPTERS_ARCH = ArchSplitConfig(
    domain="LLM_ADAPTERS",
    h2_satellite_markers=(
        ("## Model routing and failover", "routing_failover"),
        ("## Providers (19)", "providers_catalog"),
        ("## Audit register (2026-06-14)", "audit_register"),
    ),
)

RAG_ARCH = ArchSplitConfig(
    domain="RAG",
    h2_satellite_markers=(
        ("## GraphRAG architecture", "graph_rag"),
        ("## End-to-end pipelines", "pipelines_detail"),
    ),
)

SKILLS_ARCH = ArchSplitConfig(
    domain="SKILLS",
    h2_satellite_markers=(("## First-party catalog (149 skills", "skill_catalog"),),
)

MODALITY_ARCH = ArchSplitConfig(
    domain="MODALITY",
    h2_satellite_markers=(("## Tool surface", "tool_surface_detail"),),
)

CONFIGS: dict[str, ArchSplitConfig] = {
    c.domain: c
    for c in (
        ACP,
        TIER3,
        PLATFORM_ARCH,
        TOOLS_ARCH,
        UAEP_ARCH,
        NEXUS_ARCH,
        ORCH_ARCH,
        INTEGRATIONS_ARCH,
        ADAPTIVE_HARNESS_INTELLIGENCE_ARCH,
        CODE_CRAFT_ARCH,
        CONTEXT_ENGINEERING_ARCH,
        CRITIC_VERIFICATION_ARCH,
        ELASTIC_CAPACITY_AND_SCALING_ARCH,
        EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE_ARCH,
        LLM_ADAPTERS_ARCH,
        MEMORY_ARCH,
        MODALITY_ARCH,
        OBSERVABILITY_ARCH,
        RAG_ARCH,
        REASONING_AND_COGNITION_ARCH,
        RELIABILITY_FAILURE_AND_HITL_ARCH,
        SKILLS_ARCH,
    )
}
