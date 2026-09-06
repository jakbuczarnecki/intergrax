# © Artur Czarnecki. All rights reserved.

"""Project host authority state into Stage-5 capability governance evidence (composition)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.capability_catalog.adapters.tool import TOOL_BUILTIN_CATALOG_SOURCE_ID
from intergrax.capability_catalog.adapters.skill import SKILL_BUILTIN_CATALOG_SOURCE_ID
from intergrax.contracts.capability_catalog.governance import (
    CapabilityAgentGovernanceEvidence,
    CapabilitySetConstraintMode,
    CapabilitySkillGovernanceEvidence,
    CapabilityToolGovernanceEvidence,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.skills.registry.factory import enabled_skill_ids_for_profile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.tool_requirements import available_tool_ids_for_profile
from intergrax.tools.registry.profile import ToolProfile


@dataclass(frozen=True, slots=True)
class ProductionCapabilityGovernanceEvidenceBundle:
    """Typed governance evidence supplied by production composition."""

    tool_evidence: CapabilityToolGovernanceEvidence | None = None
    agent_evidence: CapabilityAgentGovernanceEvidence | None = None
    skill_evidence: CapabilitySkillGovernanceEvidence | None = None


def project_tool_governance_evidence_from_tool_profile(
    tool_profile: ToolProfile,
) -> CapabilityToolGovernanceEvidence:
    """Project host ``ToolProfile`` effective availability into tool governance evidence."""
    allowed_keys = tuple(
        CapabilityIdentityKey(
            kind=CapabilityKind.TOOL,
            source_id=TOOL_BUILTIN_CATALOG_SOURCE_ID,
            source_kind=CapabilitySourceKind.BUILTIN,
            logical_id=tool_id,
        )
        for tool_id in available_tool_ids_for_profile(tool_profile)
    )
    return CapabilityToolGovernanceEvidence(
        allowed_keys=allowed_keys,
        allowed_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
    )


def project_skill_governance_evidence_from_skill_profile(
    skill_profile: SkillProfile,
) -> CapabilitySkillGovernanceEvidence:
    """Project host ``SkillProfile`` effective enablement into skill governance evidence."""
    enabled_keys = tuple(
        CapabilityIdentityKey(
            kind=CapabilityKind.SKILL,
            source_id=SKILL_BUILTIN_CATALOG_SOURCE_ID,
            source_kind=CapabilitySourceKind.BUILTIN,
            logical_id=skill_id,
        )
        for skill_id in enabled_skill_ids_for_profile(skill_profile)
    )
    return CapabilitySkillGovernanceEvidence(
        enabled_keys=enabled_keys,
        enabled_constraint_mode=CapabilitySetConstraintMode.EXPLICIT_SET,
    )


def project_production_capability_governance_evidence(
    environment: ApplicationEnvironmentProfile,
    *,
    agent_evidence: CapabilityAgentGovernanceEvidence | None = None,
) -> ProductionCapabilityGovernanceEvidenceBundle:
    """Project host profile authority into Stage-5 governance evidence carriers."""
    return ProductionCapabilityGovernanceEvidenceBundle(
        tool_evidence=project_tool_governance_evidence_from_tool_profile(
            environment.tool_profile,
        ),
        skill_evidence=project_skill_governance_evidence_from_skill_profile(
            environment.skill_profile,
        ),
        agent_evidence=agent_evidence,
    )


__all__ = [
    "ProductionCapabilityGovernanceEvidenceBundle",
    "project_production_capability_governance_evidence",
    "project_skill_governance_evidence_from_skill_profile",
    "project_tool_governance_evidence_from_tool_profile",
]
