# © Artur Czarnecki. All rights reserved.

"""Deterministic reference Skill for ME-15 marketplace lifecycle composition proofs."""

from __future__ import annotations

from typing import Final

from intergrax.contracts.capability_catalog import CapabilityReleaseIdentity
from intergrax.skills.core.contracts import SkillManifest

ME15_SKILL_LOGICAL_ID: Final = "skills.me15.canonical-instruction"
ME15_PACKAGE_REFERENCE_V1: Final = "pkg:skill/me15-canonical-instruction"
ME15_VERSION_V1: Final = "1.0.0"
ME15_VERSION_V2: Final = "2.0.0"
ME15_DIGEST_V1: Final = "sha256:" + ("c" * 64)
ME15_DIGEST_V2: Final = "sha256:" + ("d" * 64)
ME15_INSTRUCTION_MARKER_V1: Final = "instruction.me15.v1"
ME15_INSTRUCTION_MARKER_V2: Final = "instruction.me15.v2"


def me15_manifest_for_release(release: CapabilityReleaseIdentity) -> SkillManifest:
    version = release.version_label
    digest = release.content_digest
    if version == ME15_VERSION_V2 or digest == ME15_DIGEST_V2:
        return SkillManifest(
            skill_id=ME15_SKILL_LOGICAL_ID,
            version=ME15_VERSION_V2,
            description="ME-15 reference skill v2",
            prompt_instruction_ids=(ME15_INSTRUCTION_MARKER_V2,),
        )
    if version == ME15_VERSION_V1 or digest == ME15_DIGEST_V1:
        return SkillManifest(
            skill_id=ME15_SKILL_LOGICAL_ID,
            version=ME15_VERSION_V1,
            description="ME-15 reference skill v1",
            prompt_instruction_ids=(ME15_INSTRUCTION_MARKER_V1,),
        )
    raise ValueError(
        f"unsupported ME-15 skill release version={version!r} digest={digest!r}",
    )


def instruction_marker_for_release(release: CapabilityReleaseIdentity) -> str:
    manifest = me15_manifest_for_release(release)
    if not manifest.prompt_instruction_ids:
        raise ValueError("reference skill manifest lacks instruction marker")
    return manifest.prompt_instruction_ids[0]


__all__ = [
    "ME15_DIGEST_V1",
    "ME15_DIGEST_V2",
    "ME15_INSTRUCTION_MARKER_V1",
    "ME15_INSTRUCTION_MARKER_V2",
    "ME15_PACKAGE_REFERENCE_V1",
    "ME15_SKILL_LOGICAL_ID",
    "ME15_VERSION_V1",
    "ME15_VERSION_V2",
    "instruction_marker_for_release",
    "me15_manifest_for_release",
]
