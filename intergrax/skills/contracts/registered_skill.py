# © Artur Czarnecki. All rights reserved.

"""Read-model for a skill entry on ``SkillRegistryRead``."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.skills.contracts.skill_runtime_binding_metadata import (
    SkillRuntimeBindingMetadata,
)
from intergrax.skills.core.contracts import SkillManifest


@dataclass(frozen=True, slots=True)
class RegisteredSkill:
    """Skill catalog entry returned by read-only registry contracts."""

    manifest: SkillManifest
    binding: SkillRuntimeBindingMetadata | None = None


__all__ = ["RegisteredSkill"]
