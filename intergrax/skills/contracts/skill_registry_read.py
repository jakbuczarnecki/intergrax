# © Artur Czarnecki. All rights reserved.

"""Read-only Skill registry surface for application contracts and host lifecycle."""

from __future__ import annotations

from typing import Protocol

from intergrax.skills.contracts.registered_skill import RegisteredSkill
from intergrax.skills.contracts.skill_runtime_binding_metadata import (
    SkillRuntimeBindingMetadata,
)


class SkillRegistryRead(Protocol):
    """Host-scoped skill registry read — not marketplace authority."""

    def has(self, skill_id: str) -> bool: ...

    def get(self, skill_id: str) -> RegisteredSkill: ...

    def binding_metadata(self, skill_id: str) -> SkillRuntimeBindingMetadata | None: ...


__all__ = ["SkillRegistryRead"]
