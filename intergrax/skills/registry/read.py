# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only Skill registry surface for host lifecycle and composition."""

from __future__ import annotations

from typing import Protocol

from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.runtime import RegisteredSkill, SkillRegistry


class SkillRegistryRead(Protocol):
    """Host-scoped skill registry read — not marketplace authority."""

    def has(self, skill_id: str) -> bool: ...

    def get(self, skill_id: str) -> RegisteredSkill: ...

    def binding_metadata(self, skill_id: str) -> SkillRuntimeBindingMetadata | None: ...


def as_skill_registry_read(registry: SkillRegistry) -> SkillRegistryRead:
    return registry


__all__ = ["SkillRegistryRead", "as_skill_registry_read"]
