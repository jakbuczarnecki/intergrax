# © Artur Czarnecki. All rights reserved.

"""First-party :class:`SkillPlugin` classes for all shipped skill bundles."""

from __future__ import annotations

from intergrax.skills.providers.harness.plugin import HarnessSkillPlugin
from intergrax.skills.providers.knowledge.plugin import KnowledgeSkillPlugin
from intergrax.skills.providers.legal.plugin import LegalSkillPlugin
from intergrax.skills.providers.research.plugin import ResearchSkillPlugin

SHIPPED_SKILL_PLUGINS: tuple[type, ...] = (
    HarnessSkillPlugin,
    KnowledgeSkillPlugin,
    LegalSkillPlugin,
    ResearchSkillPlugin,
)

SHIPPED_SKILL_BUNDLE_IDS: frozenset[str] = frozenset(
    p.skill_bundle_manifest().bundle_id for p in SHIPPED_SKILL_PLUGINS
)
