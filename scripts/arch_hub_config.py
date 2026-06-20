# © Artur Czarnecki. All rights reserved.
"""Per-domain architecture hub split configuration (F4 / F4-B)."""

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


ACP = ArchSplitConfig(
    domain="AGENT_CONTRACTS_AND_ASSEMBLY",
    hub_section_max=21,
    extra_hub_sections=(45,),
)

TIER3 = ArchSplitConfig(
    domain="TIER3_APPLICATION_ENVIRONMENT",
    hub_section_max=25,
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
        ("## Catalog tools", "catalog_and_index"),
        ("## Tool selection", "selection_and_plugins"),
    ),
)

UAEP_ARCH = ArchSplitConfig(
    domain="UNIFIED_EXECUTION_RUNTIME",
    subsection_major=42,
    subsection_minor_max=15,
)

NEXUS_ARCH = ArchSplitConfig(
    domain="NEXUS_EXECUTION_FLOW",
    numbered_h2_max=18,
)

ORCH_ARCH = ArchSplitConfig(
    domain="ORCHESTRATION",
    hub_section_max=26,
)

INTEGRATIONS_ARCH = ArchSplitConfig(
    domain="INTEGRATIONS",
    h2_satellite_markers=(
        ("## Full provider index", "provider_index"),
        ("## Implemented providers", "provider_catalog"),
    ),
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
    )
}
