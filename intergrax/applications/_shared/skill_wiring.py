# © Artur Czarnecki. All rights reserved.

"""Tier-3 skill catalog wiring (Phase R-Skill.4, SK-PRESET.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.skills.registry import SkillProfile, SkillRegistry, build_registry_from_profile


@dataclass(frozen=True)
class ApplicationSkillWiring:
    profile: SkillProfile
    registry: SkillRegistry


def build_application_skill_wiring(profile: SkillProfile) -> ApplicationSkillWiring:
    skill_bundle_ids = tuple(profile.enabled_bundles) if profile.enabled_bundles else None
    bootstrap_catalogs(
        register_shipped=True,
        skill_bundle_ids=skill_bundle_ids,
        discover_entry_points=discover_plugins_enabled(),
    )
    registry = build_registry_from_profile(profile)
    return ApplicationSkillWiring(profile=profile, registry=registry)


def harness_platform_skill_profile() -> SkillProfile:
    """Platform-only harness skills (no domain bundles)."""
    return SkillProfile(enabled_bundles=["harness"])


def lab_skill_profile() -> SkillProfile:
    """Lab harness preset: harness + domain + universal packs."""
    return SkillProfile(
        enabled_bundles=[
            "harness",
            "legal",
            "research",
            "rag",
            "workspace",
            "memory",
            "knowledge",
        ]
    )


def research_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["research", "rag", "browser"])


def legal_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["legal", "rag", "knowledge", "workspace"])


def knowledge_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["knowledge"])


def rag_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["rag"])


def ops_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["ops", "dev", "workspace"])


def platform_skill_profile() -> SkillProfile:
    """Intergrax assistant hub: concierge + universal retrieval packs."""
    return SkillProfile(enabled_bundles=["platform", "rag", "memory", "research"])


def lkw_skill_profile() -> SkillProfile:
    """Local Knowledge Workspace: ingest, Q&A, workspace authoring."""
    return SkillProfile(enabled_bundles=["rag", "workspace", "memory", "knowledge"])


def dispute_skill_profile() -> SkillProfile:
    """Dispute simulation: legal research, memory scratchpad, citation synthesis."""
    return SkillProfile(enabled_bundles=["legal", "rag", "memory", "research"])
