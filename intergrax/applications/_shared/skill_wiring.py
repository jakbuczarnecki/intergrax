# © Artur Czarnecki. All rights reserved.

"""Tier-3 skill catalog wiring (Phase R-Skill.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.skills.registry import SkillProfile, SkillRegistry, build_registry_from_profile
from intergrax.skills.registry.bootstrap import register_default_skills


@dataclass(frozen=True)
class ApplicationSkillWiring:
    profile: SkillProfile
    registry: SkillRegistry


def build_application_skill_wiring(profile: SkillProfile) -> ApplicationSkillWiring:
    register_default_skills()
    registry = build_registry_from_profile(profile)
    return ApplicationSkillWiring(profile=profile, registry=registry)


def lab_skill_profile() -> SkillProfile:
    """Lab may host legal mock paths and optional research agents — enable both bundles."""
    return SkillProfile(enabled_bundles=["legal", "research"])


def research_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["research"])
