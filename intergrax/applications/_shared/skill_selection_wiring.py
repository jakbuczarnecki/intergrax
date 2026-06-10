# © Artur Czarnecki. All rights reserved.

"""Dynamic skill selection L4 hook wiring (AUDIT-IDEAL-12.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.adaptive.skill_selection_engine import SkillSelectionEngine
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind


@dataclass(frozen=True, slots=True)
class SkillSelectionHook:
    enabled: bool
    engine_id: str
    candidate_bundles: tuple[str, ...]


def resolve_skill_selection_hook(env: ApplicationEnvironmentProfile) -> SkillSelectionHook:
    """Expose skill-selection proposals when adaptive routing tuning is enabled."""
    profile = env.adaptive_profile
    bundles = tuple(env.skill_profile.enabled_bundles) or ("rag", "workspace", "memory")
    loop_enabled = AdaptiveLoopKind.ROUTING_TUNING in profile.enabled_loops
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    enabled = is_product and profile.enabled and loop_enabled
    engine = SkillSelectionEngine(candidate_bundles=bundles)
    return SkillSelectionHook(
        enabled=enabled,
        engine_id=engine.engine_id,
        candidate_bundles=bundles,
    )
