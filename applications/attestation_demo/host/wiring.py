# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.lab_harness_context import lab_harness_context_from_build_context
from intergrax.applications._shared.wiring import build_manifest_development_registry
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.registry.agent_registry import AgentRegistry
from attestation_demo.host.agent_builders import build_attestation_demo_agent_builders
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
    composition = env_wiring.composition
    builders = build_attestation_demo_agent_builders(
        tool_profile=composition.tool_profile,
        lab_harness=lab_harness_context_from_build_context(
            env_wiring.build_context,
            policy_bundle=composition.policy_bundle,
            tool_wiring_context=composition.tool_wiring_context,
        ),
        boundary_event_buffer=composition.boundary_event_buffer,
    )
    return build_manifest_development_registry(
        manifest,
        env_wiring.build_context,
        builders=builders,
        composition=composition,
    )
