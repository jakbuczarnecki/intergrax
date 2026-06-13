# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.wiring import build_application_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.registry.agent_registry import AgentRegistry
from attestation_demo.host.agent_builders import ATTESTATION_DEMO_AGENT_BUILDERS
from attestation_demo.host.settings import AttestationDemoSettings
from attestation_demo.manifest import build_attestation_demo_manifest


def build_attestation_demo_registry(
    *,
    settings: AttestationDemoSettings | None = None,
) -> AgentRegistry:
    _ = settings
    manifest = build_attestation_demo_manifest()
    env = manifest.environment or ApplicationEnvironmentProfile.lab_defaults(
        profile_id="attestation_demo",
        harness_tools=False,
    ).with_reference_host_platform_defaults()
    if manifest.environment is None:
        manifest = manifest.model_copy(update={"environment": env})
    env_wiring = wire_application_environment(manifest, env)
    return build_application_registry(
        manifest,
        env_wiring.build_context,
        builders=ATTESTATION_DEMO_AGENT_BUILDERS,
    )
