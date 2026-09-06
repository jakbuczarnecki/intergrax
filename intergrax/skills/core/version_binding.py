# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill version binding contracts (CAPABILITY-CATALOG Stage 6 / SKILLS identity)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.skills.core.contracts import SkillManifest


class SkillVersionResolutionMode(str, Enum):
    """How a resolved skill version was bound during composition."""

    PINNED = "pinned"
    MATERIALIZED = "materialized"


class ResolvedSkillRole(str, Enum):
    """Whether the skill entered resolution as a root declaration or transitively."""

    ROOT = "root"
    TRANSITIVE = "transitive"


@dataclass(frozen=True, slots=True)
class ResolvedSkillRef:
    """Immutable evidence for one resolved skill in a composition snapshot."""

    skill_id: str
    version: str
    qualified_id: str
    resolution_mode: SkillVersionResolutionMode
    role: ResolvedSkillRole

    @staticmethod
    def from_manifest(
        manifest: SkillManifest,
        *,
        resolution_mode: SkillVersionResolutionMode,
        role: ResolvedSkillRole,
    ) -> ResolvedSkillRef:
        return ResolvedSkillRef(
            skill_id=manifest.skill_id,
            version=manifest.version,
            qualified_id=manifest.qualified_id,
            resolution_mode=resolution_mode,
            role=role,
        )
