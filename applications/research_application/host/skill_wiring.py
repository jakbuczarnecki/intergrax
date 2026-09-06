# © Artur Czarnecki. All rights reserved.

"""Skill catalog selection for research_application."""

from __future__ import annotations

from intergrax.skills.providers.research.manifests import RESEARCH_LITERATURE_SCAN
from intergrax.skills.registry.profile import SkillProfile

RESEARCH_BUNDLE_ID = "research"

# Default production selection aligned with ResearchAgent contract and host ToolProfile.
RESEARCH_DEFAULT_ENABLED_SKILL_IDS: tuple[str, ...] = (
    RESEARCH_LITERATURE_SCAN.skill_id,
)


def build_research_skill_profile(
    *,
    enabled_skill_ids: tuple[str, ...] | None = None,
) -> SkillProfile:
    """Register the research bundle but enable only explicitly selected skills."""
    selected = (
        enabled_skill_ids
        if enabled_skill_ids is not None
        else RESEARCH_DEFAULT_ENABLED_SKILL_IDS
    )
    return SkillProfile(
        enabled_bundles=[RESEARCH_BUNDLE_ID],
        enabled=list(selected),
    )
