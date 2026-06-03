# © Artur Czarnecki. All rights reserved.

"""Unified Tier-3 environment wiring entry (Phase H-APP.1.4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.applications._shared.environment_conformance import (
    EnvironmentSkillToolConsistencyCheck,
)
from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.applications._shared.modality_wiring import wire_modality_extras
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.sandbox_wiring import tool_profile_with_sandbox, wire_sandbox_sessions
from intergrax.applications._shared.shadow_wiring import wire_shadow_workspace
from intergrax.applications._shared.skill_wiring import ApplicationSkillWiring, build_application_skill_wiring
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.tools.registry.wiring import ToolWiringContext


@dataclass(frozen=True)
class ApplicationEnvironmentWiring:
    """Resolved environment artifacts for host factories."""

    profile: ApplicationEnvironmentProfile
    tool_wiring: ApplicationToolWiring
    skill_wiring: ApplicationSkillWiring
    policy_bundle: Any
    build_context: ApplicationBuildContext
    shadow_manager: ShadowWorkspaceManager | None
    sandbox_manager: SandboxSessionManager | None


def wire_application_environment(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    *,
    settings: Any = None,
    integration_profile: Any = None,
    runtime_event_bus: RuntimeEventBus | None = None,
    strict_harness: bool = False,
    trace_db_path: Path | None = None,
    sandbox_session: Any | None = None,
    websearch_executor: Any | None = None,
    conformance_check: bool = True,
) -> ApplicationEnvironmentWiring:
    """
    Single Tier-3 entry: catalogs, modality, policy, tool/skill registries.

    Replaces scattered per-host wiring sequences (lab/legal/research/poc).
    """
    bootstrap_application_integration_catalog()
    resolved_integration = integration_profile or env.integration_profile or manifest.integration_profile

    tool_profile = tool_profile_with_sandbox(env)
    wiring_context = ToolWiringContext.from_integration_profile(resolved_integration)
    if env.modality_profile is not None:
        wire_modality_extras(wiring_context, modality_profile=env.modality_profile)

    tool_wiring = build_application_tool_wiring(
        tool_profile,
        integration_profile=resolved_integration,
        wiring_context=wiring_context,
        sandbox_session=sandbox_session,
        websearch_executor=websearch_executor,
    )
    skill_wiring = build_application_skill_wiring(env.skill_profile)
    policy_bundle = wire_policy_bundle(env)

    tool_registry = tool_wiring.registry
    if not tool_wiring.profile.enabled and not tool_wiring.profile.enabled_bundles:
        tool_registry = None

    build_context = ApplicationBuildContext.for_manifest(
        manifest,
        settings=settings,
        integration_profile=resolved_integration,
        tool_profile=tool_wiring.profile,
        tool_wiring_context=tool_wiring.wiring_context,
        skill_profile=skill_wiring.profile,
        skill_registry=skill_wiring.registry,
        tool_registry=tool_registry,
        policy_bundle=policy_bundle,
        runtime_event_bus=runtime_event_bus or RuntimeEventBus(),
        strict_harness=strict_harness,
        trace_db_path=trace_db_path,
        environment=env,
    )

    if conformance_check:
        EnvironmentSkillToolConsistencyCheck(fail_on_violation=False).validate_roster(
            manifest.agents,
            env,
        )

    return ApplicationEnvironmentWiring(
        profile=env,
        tool_wiring=tool_wiring,
        skill_wiring=skill_wiring,
        policy_bundle=policy_bundle,
        build_context=build_context,
        shadow_manager=wire_shadow_workspace(env),
        sandbox_manager=wire_sandbox_sessions(env),
    )
