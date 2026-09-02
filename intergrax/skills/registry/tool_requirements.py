# © Artur Czarnecki. All rights reserved.

"""Resolve skill-declared tool requirements against host tool availability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.catalog import list_catalog_tool_ids
from intergrax.tools.registry.profile import ToolProfile


@dataclass(frozen=True, slots=True)
class SkillToolRequirementViolation:
    """A required tool that is unavailable on the host, with skill provenance."""

    skill_id: str
    tool_id: str


@dataclass(frozen=True, slots=True)
class SkillToolRequirementResolution:
    """Outcome of comparing skill tool requirements to host availability."""

    required_tool_ids: tuple[str, ...]
    available_tool_ids: tuple[str, ...]
    satisfied_tool_ids: tuple[str, ...]
    missing_tool_ids: tuple[str, ...]
    is_satisfied: bool
    violations: tuple[SkillToolRequirementViolation, ...] = ()


class SkillToolRequirementError(ValueError):
    """Raised when enabled skills require tools outside host availability."""

    def __init__(self, resolution: SkillToolRequirementResolution) -> None:
        self.resolution = resolution
        lines = [
            f"- {violation.skill_id} -> {violation.tool_id}"
            for violation in resolution.violations
        ]
        message = "Skill tool requirements are not satisfied:\n" + "\n".join(lines)
        super().__init__(message)


def _normalize_tool_ids(tool_ids: Collection[str]) -> frozenset[str]:
    return frozenset(tool_id.strip() for tool_id in tool_ids if tool_id.strip())


def resolve_skill_tool_requirements(
    skill_registry: SkillRegistry,
    available_tool_ids: Collection[str],
) -> SkillToolRequirementResolution:
    """
    Compare skill manifest ``tool_ids`` against host availability.

    Does not mutate the registry or any profile.
    """
    available = _normalize_tool_ids(available_tool_ids)
    required: set[str] = set()
    violations: list[SkillToolRequirementViolation] = []

    for skill_id in skill_registry.skill_ids():
        manifest = skill_registry.get(skill_id).manifest
        for tool_id in manifest.tool_ids:
            normalized = tool_id.strip()
            if not normalized:
                continue
            required.add(normalized)
            if normalized not in available:
                violations.append(
                    SkillToolRequirementViolation(skill_id=skill_id, tool_id=normalized)
                )

    missing = tuple(sorted({violation.tool_id for violation in violations}))
    required_tuple = tuple(sorted(required))
    satisfied = tuple(sorted(required & available))
    ordered_violations = tuple(
        sorted(violations, key=lambda item: (item.tool_id, item.skill_id))
    )

    return SkillToolRequirementResolution(
        required_tool_ids=required_tuple,
        available_tool_ids=tuple(sorted(available)),
        satisfied_tool_ids=satisfied,
        missing_tool_ids=missing,
        is_satisfied=not missing,
        violations=ordered_violations,
    )


def available_tool_ids_for_profile(tool_profile: ToolProfile) -> tuple[str, ...]:
    """Canonical host tool availability for composition-time validation."""
    if tool_profile.register_all_catalog_bundles:
        return tuple(sorted(list_catalog_tool_ids()))

    available = {tool_id.strip() for tool_id in tool_profile.enabled if tool_id.strip()}
    for tool_id in list_catalog_tool_ids():
        if tool_profile.is_tool_enabled(tool_id):
            available.add(tool_id)
    return tuple(sorted(available))


def assert_skill_tool_requirements_satisfied(
    skill_registry: SkillRegistry,
    tool_profile: ToolProfile,
) -> SkillToolRequirementResolution:
    """Fail closed when any enabled skill requires unavailable host tools."""
    resolution = resolve_skill_tool_requirements(
        skill_registry,
        available_tool_ids_for_profile(tool_profile),
    )
    if not resolution.is_satisfied:
        raise SkillToolRequirementError(resolution)
    return resolution
