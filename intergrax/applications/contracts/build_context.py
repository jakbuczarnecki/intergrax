# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime context passed to Tier-3 agent factories (Phase N.2.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True)
class ApplicationBuildContext:
    """
    Inputs available when materializing agents for an application host.

    ``settings`` is application-specific (e.g. ``LabApplicationSettings``,
    ``LegalBackendSettings``). Factories read env-backed settings here — not
    from global process env directly.
    """

    manifest: Any
    settings: Any = None
    integration_profile: IntegrationProfile | None = None
    tool_profile: ToolProfile | None = None
    tool_wiring_context: ToolWiringContext | None = None
    skill_profile: SkillProfile | None = None
    skill_registry: SkillRegistry | None = None
    tool_registry: ToolRegistry | None = None

    @classmethod
    def for_manifest(
        cls,
        manifest: Any,
        *,
        settings: Any = None,
        tool_profile: ToolProfile | None = None,
        tool_wiring_context: ToolWiringContext | None = None,
        skill_profile: SkillProfile | None = None,
        skill_registry: SkillRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> ApplicationBuildContext:
        profile = getattr(manifest, "integration_profile", None)
        return cls(
            manifest=manifest,
            settings=settings,
            integration_profile=profile,
            tool_profile=tool_profile,
            tool_wiring_context=tool_wiring_context,
            skill_profile=skill_profile,
            skill_registry=skill_registry,
            tool_registry=tool_registry,
        )
