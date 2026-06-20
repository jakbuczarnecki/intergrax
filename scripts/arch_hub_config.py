# © Artur Czarnecki. All rights reserved.
"""Per-domain architecture hub split configuration (F4)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ArchSplitConfig:
    domain: str
    hub_section_max: int
    production_section_min: int = 40
    extra_hub_sections: tuple[int, ...] = ()
    h2_satellite_markers: tuple[tuple[str, str], ...] = field(default_factory=tuple)


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
    hub_section_max=0,
    h2_satellite_markers=(
        ("## Tool selection", "selection_and_plugins"),
        ("## Catalog tools", "catalog_and_index"),
    ),
)

CONFIGS: dict[str, ArchSplitConfig] = {
    c.domain: c
    for c in (ACP, TIER3, PLATFORM_ARCH, TOOLS_ARCH)
}
