# © Artur Czarnecki. All rights reserved.

"""Unified Tier-3 environment wiring entry (Phase H-APP.1.4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.applications._shared.environment_conformance import (
    EnvironmentSkillToolConsistencyCheck,
)
from intergrax.applications._shared.integration_health_wiring import probe_integration_profile_health
from intergrax.applications._shared.integration_wiring import bootstrap_application_integration_catalog
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications._shared.rag_runtime_bridge import resolve_rag_stack_for_environment
from intergrax.applications._shared.modality_wiring import wire_modality_extras
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications._shared.prompt_wiring import resolve_prompt_registry
from intergrax.applications._shared.capability_graph_assembly_resolver import (
    assert_capability_graph_assembly_valid,
)
from intergrax.applications._shared.capability_graph_wiring import (
    EnvironmentCapabilityGraphView,
    resolve_environment_capability_graph,
)
from intergrax.applications._shared.reliability_wiring import wire_application_reliability
from intergrax.applications._shared.registry_assembly_resolver import assert_registry_assembly_valid
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot, resolve_registry_snapshot
from intergrax.applications._shared.sandbox_wiring import tool_profile_with_sandbox, wire_sandbox_sessions
from intergrax.applications._shared.shadow_wiring import wire_shadow_workspace
from intergrax.applications._shared.skill_wiring import ApplicationSkillWiring, build_application_skill_wiring
from intergrax.applications._shared.tool_wiring import ApplicationToolWiring, build_application_tool_wiring
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
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
    integration_health: tuple[HealthStatus, ...] = ()
    prompt_registry: YamlPromptRegistry | None = None
    registry_snapshot: HarnessRegistrySnapshot | None = None
    capability_graph: EnvironmentCapabilityGraphView | None = None


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
    reliability_wiring = wire_application_reliability(env)
    integration_health = probe_integration_profile_health(
        resolved_integration,
        circuit_breaker_config=reliability_wiring.circuit_breaker_config,
    )

    rag_stack = resolve_rag_stack_for_environment(
        env,
        integration_profile=resolved_integration,
        llm_adapter=resolve_llm_adapter(env),
    )

    tool_profile = tool_profile_with_sandbox(env)
    wiring_context = ToolWiringContext.from_integration_profile(resolved_integration)
    if env.modality_profile is not None:
        wire_modality_extras(wiring_context, modality_profile=env.modality_profile)
    from intergrax.applications._shared.integration_tool_wiring import wire_integration_tool_context

    wiring_context = wire_integration_tool_context(wiring_context, resolved_integration)

    hosted_session = None
    if wiring_context.sandbox_session is None and resolved_integration is not None:
        from intergrax.applications._shared.sandbox_host_wiring import resolve_hosted_sandbox_session

        hosted_session = resolve_hosted_sandbox_session(
            resolved_integration,
            tenant_id="harness",
            task_id="bootstrap",
        )

    tool_wiring = build_application_tool_wiring(
        tool_profile,
        integration_profile=resolved_integration,
        wiring_context=wiring_context,
        vectorstore_manager=rag_stack.vectorstore_manager if rag_stack is not None else None,
        embedding_manager=rag_stack.embedding_manager if rag_stack is not None else None,
        retriever_manager=rag_stack.retriever_manager if rag_stack is not None else None,
        reranker_manager=rag_stack.reranker_manager if rag_stack is not None else None,
        rag_profile=rag_stack.profile if rag_stack is not None else None,
        retrieval_service=rag_stack.retrieval_service if rag_stack is not None else None,
        sandbox_session=sandbox_session or hosted_session,
        websearch_executor=websearch_executor,
    )
    skill_wiring = build_application_skill_wiring(env.skill_profile)
    policy_bundle = wire_policy_bundle(env)
    prompt_registry = resolve_prompt_registry(env.prompt_profile)

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
        prompt_registry=prompt_registry,
    )

    registry_snapshot = resolve_registry_snapshot(build_context)
    capability_graph = resolve_environment_capability_graph(manifest, env, registry_snapshot)

    if conformance_check:
        assert_registry_assembly_valid(registry_snapshot, env)
        assert_capability_graph_assembly_valid(capability_graph, registry_snapshot, manifest)
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
        integration_health=integration_health,
        prompt_registry=prompt_registry,
        registry_snapshot=registry_snapshot,
        capability_graph=capability_graph,
    )
