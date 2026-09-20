# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compatibility re-export — canonical: skills.contracts.skill_registry_read."""

from __future__ import annotations

from intergrax.skills.contracts.skill_registry_read import SkillRegistryRead
from intergrax.skills.registry.runtime import SkillRegistry


def as_skill_registry_read(registry: SkillRegistry) -> SkillRegistryRead:
    return registry


__all__ = ["SkillRegistryRead", "as_skill_registry_read"]
