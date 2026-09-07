# © Artur Czarnecki. All rights reserved.

"""Dispute sim test registry projection helper (AC-3)."""

from __future__ import annotations

from intergrax.applications._shared.production_platform_persistence import (
    resolve_reference_production_strict_host_environment,
)
from dispute_sim_application.host.agent_builders import DISPUTE_SIM_AGENT_BUILDERS
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.host.settings import DisputeSimBackendSettings
from dispute_sim_application.manifest import build_dispute_sim_manifest
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.skills.registry.profile import SkillProfile
from tests.unit.applications.ac3_projection_helpers import build_test_registry_projection


def build_dispute_sim_host_test_environment(
    settings: DisputeSimBackendSettings | None = None,
    *,
    strict_platform_backing: bool = False,
) -> ApplicationEnvironmentProfile:
    resolved_settings = settings or DisputeSimBackendSettings.from_env()
    env = build_dispute_sim_environment_profile(resolved_settings).model_copy(
        update={"skill_profile": SkillProfile()},
    )
    if strict_platform_backing and env.execution_mode.value == "strict":
        env = resolve_reference_production_strict_host_environment(env)
    return env


def build_dispute_sim_host_test_manifest(
    settings: DisputeSimBackendSettings | None = None,
    *,
    strict_platform_backing: bool = False,
) -> ApplicationManifest:
    resolved_settings = settings or DisputeSimBackendSettings.from_env()
    return build_dispute_sim_manifest().model_copy(
        update={
            "environment": build_dispute_sim_host_test_environment(
                resolved_settings,
                strict_platform_backing=strict_platform_backing,
            ),
        },
    )


def build_dispute_sim_test_registry_projection(
    *,
    revision_id: str = "dispute-sim-test-revision",
) -> MaterializedRegistryProjection:
    manifest = build_dispute_sim_host_test_manifest()
    env = manifest.environment
    return build_test_registry_projection(
        manifest,
        env,
        builders=DISPUTE_SIM_AGENT_BUILDERS,
        revision_id=revision_id,
    )
