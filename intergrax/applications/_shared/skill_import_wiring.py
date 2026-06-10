# © Artur Czarnecki. All rights reserved.

"""LangGraph-compatible skill pack import wiring (AUDIT-IDEAL-12.1)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.importers.langgraph_skill_pack import LangGraphSkillPackImporter


def resolve_langgraph_skill_import_enabled(env: ApplicationEnvironmentProfile) -> bool:
    """Product and lab hosts may import LangGraph-compatible skill packs."""
    return env.application_profile in (ApplicationProfile.PRODUCT, ApplicationProfile.LAB)


def import_langgraph_skill_pack(path: Path) -> SkillManifest:
    """Import a LangGraph JSON pack when host profile allows skill import."""
    return LangGraphSkillPackImporter().import_file(path)
