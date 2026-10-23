# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Dict, List

from intergrax.skills.contracts.registered_skill import RegisteredSkill
from intergrax.skills.contracts.skill_runtime_binding_metadata import (
    SkillRuntimeBindingMetadata,
)
from intergrax.skills.core.contracts import SkillManifest


class SkillRegistry:
    """Runtime-owned skill catalog keyed by ``skill_id`` (latest registration wins per id)."""

    def __init__(self) -> None:
        self._skills: Dict[str, RegisteredSkill] = {}

    def register(
        self,
        manifest: SkillManifest,
        *,
        binding: SkillRuntimeBindingMetadata | None = None,
    ) -> None:
        skill_id = manifest.skill_id
        if skill_id in self._skills:
            existing = self._skills[skill_id].manifest
            if existing.version != manifest.version:
                raise ValueError(
                    f"Skill '{skill_id}' already registered at version {existing.version}; "
                    f"use override flow to replace"
                )
            raise ValueError(f"Skill already registered: {skill_id}")
        self._skills[skill_id] = RegisteredSkill(manifest=manifest, binding=binding)

    def register_or_replace(
        self,
        manifest: SkillManifest,
        *,
        binding: SkillRuntimeBindingMetadata | None = None,
    ) -> None:
        self._skills[manifest.skill_id] = RegisteredSkill(
            manifest=manifest,
            binding=binding,
        )

    def get(self, skill_id: str) -> RegisteredSkill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"Skill not registered: {skill_id}") from exc

    def has(self, skill_id: str) -> bool:
        return skill_id in self._skills

    def binding_metadata(self, skill_id: str) -> SkillRuntimeBindingMetadata | None:
        registered = self._skills.get(skill_id)
        if registered is None:
            return None
        return registered.binding

    def list(self) -> List[RegisteredSkill]:
        return list(self._skills.values())

    def skill_ids(self) -> List[str]:
        return sorted(self._skills)

    def unregister(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)
