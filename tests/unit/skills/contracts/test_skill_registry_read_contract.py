# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.skills.contracts.registered_skill import RegisteredSkill
from intergrax.skills.contracts.skill_registry_read import SkillRegistryRead
from intergrax.skills.contracts.skill_runtime_binding_metadata import (
    SkillRuntimeBindingMetadata,
)
from intergrax.skills.core.contracts import SkillManifest

pytestmark = pytest.mark.unit


class _ExternalSkillRegistryRead:
    def __init__(self, manifest: SkillManifest) -> None:
        self._manifest = manifest
        self._binding: SkillRuntimeBindingMetadata | None = None

    def has(self, skill_id: str) -> bool:
        return skill_id == self._manifest.skill_id

    def get(self, skill_id: str) -> RegisteredSkill:
        if skill_id != self._manifest.skill_id:
            raise KeyError(skill_id)
        return RegisteredSkill(manifest=self._manifest, binding=self._binding)

    def binding_metadata(self, skill_id: str) -> SkillRuntimeBindingMetadata | None:
        if skill_id != self._manifest.skill_id:
            return None
        return self._binding


def test_external_structural_skill_registry_read() -> None:
    manifest = SkillManifest(skill_id="demo.skill", description="demo")
    reader: SkillRegistryRead = _ExternalSkillRegistryRead(manifest)
    assert reader.has("demo.skill")
    entry = reader.get("demo.skill")
    assert isinstance(entry, RegisteredSkill)
    assert entry.manifest.skill_id == "demo.skill"
    assert reader.binding_metadata("demo.skill") is None
