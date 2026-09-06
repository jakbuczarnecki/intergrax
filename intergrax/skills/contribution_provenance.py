# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill contribution lineage projection from resolved composition (P1.10)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.core.version_binding import (
    ResolvedSkillRef,
    ResolvedSkillRole,
    SkillVersionResolutionMode,
)
from intergrax.skills.resolver import ResolvedSkillPack, SkillResolutionError


class SkillContributionKind(str, Enum):
    """Contribution categories introduced by resolved skills."""

    TOOL_REQUIREMENT = "tool_requirement"
    PROMPT_INSTRUCTION = "prompt_instruction"
    POLICY_FRAGMENT = "policy_fragment"


@dataclass(frozen=True, slots=True)
class SkillContributionProvenance:
    """Immutable lineage for one skill-derived contribution."""

    skill_id: str
    version: str
    qualified_id: str
    role: ResolvedSkillRole
    resolution_mode: SkillVersionResolutionMode
    contribution_kind: SkillContributionKind
    contribution_id: str


def _provenance_for_ref(
    ref: ResolvedSkillRef,
    *,
    contribution_kind: SkillContributionKind,
    contribution_id: str,
) -> SkillContributionProvenance:
    return SkillContributionProvenance(
        skill_id=ref.skill_id,
        version=ref.version,
        qualified_id=ref.qualified_id,
        role=ref.role,
        resolution_mode=ref.resolution_mode,
        contribution_kind=contribution_kind,
        contribution_id=contribution_id,
    )


def _assert_manifest_matches_ref(ref: ResolvedSkillRef, manifest: SkillManifest) -> None:
    if manifest.skill_id != ref.skill_id or manifest.version != ref.version:
        raise SkillResolutionError(
            f"manifest identity mismatch for {ref.qualified_id}: "
            f"observed {manifest.skill_id}@{manifest.version}",
        )


def build_skill_contribution_provenance(
    pack: ResolvedSkillPack,
    manifests: Mapping[str, SkillManifest],
) -> tuple[SkillContributionProvenance, ...]:
    """Project per-contribution lineage from bound pack evidence and manifests."""
    entries: list[SkillContributionProvenance] = []
    ref_by_skill_id = {ref.skill_id: ref for ref in pack.resolved_skills}
    for skill_id in (ref.skill_id for ref in pack.resolved_skills):
        manifest = manifests.get(skill_id)
        if manifest is None:
            continue
        ref = ref_by_skill_id[skill_id]
        _assert_manifest_matches_ref(ref, manifest)
        for tool_id in manifest.tool_ids:
            normalized = tool_id.strip()
            if normalized:
                entries.append(
                    _provenance_for_ref(
                        ref,
                        contribution_kind=SkillContributionKind.TOOL_REQUIREMENT,
                        contribution_id=normalized,
                    ),
                )
        for prompt_id in manifest.prompt_instruction_ids:
            normalized = prompt_id.strip()
            if normalized:
                entries.append(
                    _provenance_for_ref(
                        ref,
                        contribution_kind=SkillContributionKind.PROMPT_INSTRUCTION,
                        contribution_id=normalized,
                    ),
                )
        if manifest.policy_fragment_id:
            fragment_id = manifest.policy_fragment_id.strip()
            if fragment_id:
                entries.append(
                    _provenance_for_ref(
                        ref,
                        contribution_kind=SkillContributionKind.POLICY_FRAGMENT,
                        contribution_id=fragment_id,
                    ),
                )

    return tuple(
        sorted(
            entries,
            key=lambda item: (
                item.contribution_kind.value,
                item.contribution_id,
                item.qualified_id,
            ),
        ),
    )


def contributors_for(
    provenance: tuple[SkillContributionProvenance, ...],
    *,
    contribution_kind: SkillContributionKind,
    contribution_id: str,
) -> tuple[str, ...]:
    """Return sorted qualified skill ids that introduced one contribution."""
    normalized = contribution_id.strip()
    return tuple(
        sorted(
            {
                item.qualified_id
                for item in provenance
                if item.contribution_kind is contribution_kind
                and item.contribution_id == normalized
            },
        ),
    )


def highest_risk_contributors(
    pack: ResolvedSkillPack,
    manifests: Mapping[str, SkillManifest],
) -> tuple[str, ...]:
    """Return qualified ids of skills that introduced the resolved pack max risk tier."""
    if not pack.resolved_skills:
        return ()
    risk_order = list(manifests[ref.skill_id].risk_tier for ref in pack.resolved_skills if ref.skill_id in manifests)
    if not risk_order:
        return ()
    max_tier = pack.risk_tier
    return tuple(
        sorted(
            ref.qualified_id
            for ref in pack.resolved_skills
            if ref.skill_id in manifests and manifests[ref.skill_id].risk_tier is max_tier
        ),
    )
