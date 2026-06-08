# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.skills.core.manifest import SkillBundleManifest
from intergrax.skills.providers.modality.manifests import (
    MODALITY_SPEECH_IO,
    MODALITY_VISION_OCR,
    MODALITY_VISION_SEGMENT,
    MODALITY_AUDIO_TRANSCRIPT,
    MODALITY_IMAGE_ANALYST,
)
from intergrax.skills.registry.catalog import SkillBundleStatus
from intergrax.skills.registry.runtime import SkillRegistry

_MODALITY_MANIFESTS = (
    MODALITY_SPEECH_IO,
    MODALITY_VISION_OCR,
    MODALITY_VISION_SEGMENT,
    MODALITY_AUDIO_TRANSCRIPT,
    MODALITY_IMAGE_ANALYST,
)


class ModalitySkillPlugin:
    @classmethod
    def skill_bundle_manifest(cls) -> SkillBundleManifest:
        return SkillBundleManifest(
            bundle_id="modality",
            skill_ids=tuple(m.skill_id for m in _MODALITY_MANIFESTS),
            status=SkillBundleStatus.STABLE,
            description="modality skill packs (SK-EXP5)",
        )

    @classmethod
    def skill_manifests(cls) -> tuple:
        return _MODALITY_MANIFESTS

    @classmethod
    def register_skills(cls, registry: SkillRegistry) -> None:
        for manifest in _MODALITY_MANIFESTS:
            registry.register(manifest)
