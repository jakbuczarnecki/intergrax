# © Artur Czarnecki. All rights reserved.

"""ME-16-C1 application_skill_registry → host skill wiring seam."""

from __future__ import annotations

from intergrax.agents.persistence.skill_host_wiring import (
    HostSkillCatalogWiring,
    resolve_skill_host_wiring_from_metadata,
)
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.skill_host_execution_wiring import (
    build_host_skill_catalog_wiring_from_environment,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from testing_support.canonical_agent_lifecycle_composition import _stage15_proof_environment
from testing_support.canonical_me16_mixed_agent import ME16_MIXED_CONTRACT_ID


def test_application_skill_registry_builds_host_skill_catalog_wiring() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-profile-me16-c1-unit")
    manifest = ApplicationManifest.lab(
        app_id="me16_c1_skill_seam",
        name="Skill seam",
        route_prefix="/v1/me16c1",
        env_prefix="ME16C1_",
        agents=[
            AgentBinding(
                contract_id=ME16_MIXED_CONTRACT_ID,
                builder_key=ME16_MIXED_CONTRACT_ID,
            ),
        ],
    )
    env = _stage15_proof_environment("env-me16-c1-skill")
    env = env.model_copy(update={"skill_profile": SkillProfile(enabled=[])})
    env_wiring = wire_application_environment(
        manifest,
        env,
        application_skill_registry=lifecycle.registry,
    )
    assert isinstance(lifecycle.registry, SkillRegistry)
    wiring = build_host_skill_catalog_wiring_from_environment(env_wiring)
    assert isinstance(wiring, HostSkillCatalogWiring)
    assert wiring.skill_registry is lifecycle.registry
    assert wiring.skill_profile == env_wiring.skill_wiring.profile
    metadata = {AcpMetadataKey.SKILL_HOST_WIRING: wiring}
    resolved = resolve_skill_host_wiring_from_metadata(metadata)
    assert resolved is not None
    assert resolved.skill_registry is lifecycle.registry
