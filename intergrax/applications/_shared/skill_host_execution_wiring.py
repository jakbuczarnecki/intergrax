# © Artur Czarnecki. All rights reserved.

"""Canonical host skill catalog wiring from application environment composition."""

from __future__ import annotations

from intergrax.agents.persistence.skill_host_wiring import HostSkillCatalogWiring
from intergrax.applications._shared.environment_wiring import ApplicationEnvironmentWiring


def build_host_skill_catalog_wiring_from_environment(
    env_wiring: ApplicationEnvironmentWiring,
) -> HostSkillCatalogWiring:
    """Build read-only skill host wiring carrier from wired application environment."""
    return HostSkillCatalogWiring(
        skill_profile=env_wiring.skill_wiring.profile,
        skill_registry=env_wiring.skill_wiring.registry,
        skill_pinning_store=env_wiring.build_context.skill_pinning_store,
    )


__all__ = ["build_host_skill_catalog_wiring_from_environment"]
