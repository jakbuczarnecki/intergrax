# © Artur Czarnecki. All rights reserved.

"""Skill → Tool domain dependency provider (P1.3 adoption)."""

from __future__ import annotations

from intergrax.applications.contracts.capability_dependency import (
    CapabilityDependency,
    CapabilityDependencyAvailabilityStatus,
    CapabilityDependencyKind,
    CapabilityDependencyProvider,
    CapabilityDependencyRequirement,
    CapabilityDependencyValidationContext,
    CapabilityRef,
)
from intergrax.skills.registry.factory import build_registry_from_profile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.registry.tool_requirements import (
    available_tool_ids_for_profile,
    resolve_skill_tool_requirements,
)


class SkillToolCapabilityDependencyProvider:
    """Declare skill manifest tool requirements without expanding host ToolProfile."""

    @property
    def provider_id(self) -> str:
        return "skill_tool_contract"

    @property
    def source_domain(self) -> str:
        return "skill_tool_contract"

    def dependencies_for(
        self,
        context: CapabilityDependencyValidationContext,
    ) -> tuple[CapabilityDependency, ...]:
        skill_profile = context.environment_profile.skill_profile
        if not _skill_profile_has_selection(skill_profile):
            return ()

        registry = context.skill_registry or build_registry_from_profile(skill_profile)
        declarations: list[CapabilityDependency] = []
        for skill_id in registry.skill_ids():
            manifest = registry.get(skill_id).manifest
            owner = CapabilityRef(
                kind=CapabilityDependencyKind.SKILL,
                capability_id=skill_id,
            )
            for tool_id in manifest.tool_ids:
                normalized = tool_id.strip()
                if not normalized:
                    continue
                declarations.append(
                    CapabilityDependency(
                        owner=owner,
                        dependency=CapabilityRef(
                            kind=CapabilityDependencyKind.TOOL,
                            capability_id=normalized,
                        ),
                        requirement=CapabilityDependencyRequirement.REQUIRED,
                        source_domains=(self.source_domain,),
                    ),
                )
        return tuple(
            sorted(declarations, key=lambda item: item.dedup_key),
        )

    def evaluate_availability(
        self,
        dependency: CapabilityDependency,
        context: CapabilityDependencyValidationContext,
    ) -> tuple[CapabilityDependencyAvailabilityStatus, str]:
        if self.source_domain not in dependency.source_domains:
            return (
                CapabilityDependencyAvailabilityStatus.UNKNOWN,
                f"provider cannot evaluate source domains {dependency.source_domains!r}",
            )
        if dependency.dependency.kind is not CapabilityDependencyKind.TOOL:
            return (
                CapabilityDependencyAvailabilityStatus.UNKNOWN,
                f"unsupported dependency kind {dependency.dependency.kind.value}",
            )

        available = set(
            available_tool_ids_for_profile(context.environment_profile.tool_profile),
        )
        tool_id = dependency.dependency.capability_id
        if tool_id in available:
            return (
                CapabilityDependencyAvailabilityStatus.AVAILABLE,
                "tool is effectively available on host ToolProfile",
            )
        return (
            CapabilityDependencyAvailabilityStatus.UNAVAILABLE,
            f"tool {tool_id!r} is not effectively available on host ToolProfile",
        )


def skill_tool_dependency_resolution(
    context: CapabilityDependencyValidationContext,
) -> object:
    """Expose canonical SkillToolRequirementResolution for compatibility callers."""
    skill_profile = context.environment_profile.skill_profile
    if not _skill_profile_has_selection(skill_profile):
        return resolve_skill_tool_requirements(
            build_registry_from_profile(SkillProfile()),
            available_tool_ids_for_profile(context.environment_profile.tool_profile),
        )
    registry = context.skill_registry or build_registry_from_profile(skill_profile)
    return resolve_skill_tool_requirements(
        registry,
        available_tool_ids_for_profile(context.environment_profile.tool_profile),
    )


def _skill_profile_has_selection(skill_profile: SkillProfile) -> bool:
    return bool(
        skill_profile.enabled
        or skill_profile.enabled_bundles
        or skill_profile.register_all_catalog_bundles
    )
