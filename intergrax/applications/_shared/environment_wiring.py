# © Artur Czarnecki. All rights reserved.

"""Unified Tier-3 environment wiring entry (Phase H-APP.1.4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intergrax.applications._shared.environment_conformance import (
    EnvironmentSkillToolConsistencyCheck,
    ProfileInvariantValidator,
)
from intergrax.applications._shared.integration_health_wiring import (
    probe_integration_profile_health,
)
from intergrax.applications._shared.context_wiring import (
    assert_strict_context_bootstrap_acceptable,
    bootstrap_application_context_catalog,
)
from intergrax.applications._shared.integration_wiring import (
    bootstrap_application_integration_catalog,
)
from intergrax.applications._shared.llm_resolver import resolve_environment_llm_adapter
from intergrax.applications._shared.rag_runtime_bridge import (
    resolve_rag_profile_for_environment,
    resolve_rag_stack_for_environment,
)
from intergrax.applications._shared.modality_wiring import wire_modality_extras
from intergrax.applications._shared.policy_wiring import (
    assert_strict_policy_bootstrap_acceptable,
    wire_policy_bundle,
)
from intergrax.applications._shared.prompt_wiring import resolve_prompt_registry
from intergrax.applications._shared.capability_graph_assembly_resolver import (
    assert_capability_graph_assembly_valid,
)
from intergrax.applications._shared.capability_graph_wiring import (
    EnvironmentCapabilityGraphView,
    resolve_environment_capability_graph,
)
from intergrax.applications._shared.reliability_wiring import (
    wire_application_reliability,
)
from intergrax.applications._shared.registry_assembly_resolver import (
    assert_registry_assembly_valid,
)
from intergrax.applications._shared.registry_snapshot import (
    HarnessRegistrySnapshot,
    resolve_registry_snapshot,
)
from intergrax.applications._shared.integration_tool_profile import (
    extend_tool_profile_for_integration,
)
from intergrax.applications._shared.memory_wiring import (
    assert_strict_memory_bootstrap_acceptable,
    resolve_memory_platform_wiring,
)
from intergrax.applications._shared.memory_vector_wiring import (
    assert_memory_vector_backend_available,
    build_user_profile_manager,
    resolve_rag_stack_for_memory_wiring,
)
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_manager,
)
from intergrax.applications._shared.notify_tool_wiring import (
    wire_scheduled_notification_tool_binding,
)
from intergrax.applications._shared.session_tool_wiring import (
    wire_session_storage_tool_binding,
)
from intergrax.applications._shared.capability_dependency import (
    validate_capability_dependencies_for_environment,
)
from intergrax.applications._shared.sandbox_wiring import (
    tool_profile_with_sandbox,
    wire_sandbox_sessions,
)
from intergrax.applications._shared.codecraft_wiring import (
    apply_codecraft_to_wiring_context,
    tool_profile_with_codecraft,
    wire_application_codecraft,
)
from intergrax.applications._shared.shadow_wiring import wire_shadow_workspace
from intergrax.applications._shared.skill_wiring import (
    ApplicationSkillWiring,
    assert_strict_skill_bootstrap_acceptable,
    build_application_skill_wiring,
)
from intergrax.applications._shared.application_owned_tool_conformance import (
    assert_application_owned_tool_conformance,
    platform_reserved_tool_ids,
)
from intergrax.applications._shared.application_owned_tool_wiring import (
    apply_application_owned_tool_registry,
)
from intergrax.applications._shared.tool_wiring import (
    ApplicationToolWiring,
    assert_strict_tool_bootstrap_acceptable,
    build_application_tool_wiring,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications._shared.security_assembly_resolver import SecurityAssemblyError
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.platform_plugin_evidence import (
    ApplicationPlatformPluginEvidence,
    build_application_platform_plugin_evidence,
)
from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.security_bootstrap import SecurityBootstrapResult, bootstrap_security_providers
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.core.plugins.platform_qualification import (
    PlatformPluginPackageQualificationBundle,
)
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
    platform_plugin_evidence: ApplicationPlatformPluginEvidence
    shadow_manager: ShadowWorkspaceManager | None
    sandbox_manager: SandboxSessionManager | None
    integration_health: tuple[HealthStatus, ...] = ()
    prompt_registry: YamlPromptRegistry | None = None
    registry_snapshot: HarnessRegistrySnapshot | None = None
    capability_graph: EnvironmentCapabilityGraphView | None = None


def _security_plugin_bootstrap_errors(report: DomainPluginLoadReport) -> tuple[str, ...]:
    errors: list[str] = []
    for item in report.failed:
        errors.append(f"security plugin load failed: {item.spec.name}: {item.error}")
    for item in report.rejected:
        if item.fail_closed:
            errors.append(
                "security plugin admission rejected: "
                f"{item.spec.name}: {item.reason_code.value}",
            )
    if not errors:
        errors.append("security plugin bootstrap admission is not acceptable")
    return tuple(errors)


def _bootstrap_application_security_providers() -> SecurityBootstrapResult:
    return bootstrap_security_providers(discover_entry_points=discover_plugins_enabled())


def _assert_strict_security_bootstrap_acceptable(
    env: ApplicationEnvironmentProfile,
    security_result: SecurityBootstrapResult,
) -> None:
    if env.execution_mode is not ExecutionMode.STRICT:
        return
    if security_result.critical_bootstrap_acceptable:
        return
    raise SecurityAssemblyError(_security_plugin_bootstrap_errors(security_result.load_report))


def _merge_integration_read_allowlist_roots(
    wiring_context: ToolWiringContext,
    resolved_integration: Any,
) -> ToolWiringContext:
    """Union integration option ``allowed_read_roots`` with existing context roots."""
    merged_roots = set(wiring_context.read_allowlist_roots or ())
    if resolved_integration is not None:
        for option_block in resolved_integration.options.values():
            raw_roots = option_block.get("allowed_read_roots")
            if not raw_roots:
                continue
            if isinstance(raw_roots, (str, bytes)):
                continue
            merged_roots.update(str(root) for root in raw_roots)
    if merged_roots == set(wiring_context.read_allowlist_roots or ()):
        return wiring_context
    from dataclasses import replace

    return replace(wiring_context, read_allowlist_roots=frozenset(merged_roots))


def wire_application_environment(
    manifest: ApplicationManifest,
    env: ApplicationEnvironmentProfile,
    *,
    settings: Any = None,
    integration_profile: Any = None,
    tenant_id: str | None = None,
    runtime_event_bus: RuntimeEventBus | None = None,
    strict_harness: bool = False,
    trace_db_path: Path | None = None,
    sandbox_session: Any | None = None,
    websearch_executor: Any | None = None,
    conformance_check: bool = True,
    application_tool_registry: ToolRegistry | None = None,
    document_store: Any | None = None,
    key_value_cache: Any | None = None,
    boundary_event_buffer: Any | None = None,
    platform_plugin_package_qualifications: (
        PlatformPluginPackageQualificationBundle | None
    ) = None,
) -> ApplicationEnvironmentWiring:
    """
    Single Tier-3 entry: catalogs, modality, policy, tool/skill registries.

    Replaces scattered per-host wiring sequences (lab/legal/research/poc).
    """
    bootstrap_application_integration_catalog()
    security_bootstrap = _bootstrap_application_security_providers()
    _assert_strict_security_bootstrap_acceptable(env, security_bootstrap)
    context_bootstrap = bootstrap_application_context_catalog()
    assert_strict_context_bootstrap_acceptable(env, context_bootstrap)
    resolved_integration = (
        integration_profile or env.integration_profile or manifest.integration_profile
    )
    reliability_wiring = wire_application_reliability(env)
    integration_health = probe_integration_profile_health(
        resolved_integration,
        circuit_breaker_config=reliability_wiring.circuit_breaker_config,
    )

    rag_stack = None
    host_embedding_manager = None
    host_rag_profile = None
    if env.context_profile.enable_rag:
        host_rag_profile = resolve_rag_profile_for_environment(
            env,
            integration_profile=resolved_integration,
        )
        if tenant_id is None:
            host_embedding_manager = create_default_embedding_manager()
    if tenant_id is not None:
        rag_stack = resolve_rag_stack_for_memory_wiring(
            env,
            tenant_id=tenant_id,
            integration_profile=resolved_integration,
            llm_adapter=resolve_environment_llm_adapter(env),
        )
        assert_memory_vector_backend_available(env, rag_stack)

    tool_profile = tool_profile_with_sandbox(env)
    env_for_codecraft = env.model_copy(update={"tool_profile": tool_profile})
    tool_profile = tool_profile_with_codecraft(env_for_codecraft)
    if resolved_integration is not None:
        tool_profile = extend_tool_profile_for_integration(
            tool_profile, resolved_integration
        )

    skill_bundle_ids = (
        tuple(env.skill_profile.enabled_bundles)
        if env.skill_profile.enabled_bundles
        else None
    )
    catalog_bootstrap = bootstrap_catalogs(
        register_shipped=True,
        skill_bundle_ids=skill_bundle_ids,
        discover_entry_points=discover_plugins_enabled(),
    )
    assert_strict_tool_bootstrap_acceptable(env, catalog_bootstrap.tool_plugin_load_report)
    assert_strict_skill_bootstrap_acceptable(env, catalog_bootstrap.skill_plugin_load_report)

    skill_wiring = build_application_skill_wiring(
        env.skill_profile,
        catalog_bootstrap=catalog_bootstrap,
    )

    validate_capability_dependencies_for_environment(
        env.model_copy(update={"tool_profile": tool_profile}),
        skill_registry=skill_wiring.registry,
    )
    wiring_context = ToolWiringContext.from_integration_profile(resolved_integration)
    if env.modality_profile is not None:
        wire_modality_extras(wiring_context, modality_profile=env.modality_profile)
    from intergrax.applications._shared.integration_tool_wiring import (
        wire_integration_tool_context,
    )

    wiring_context = wire_integration_tool_context(wiring_context, resolved_integration)
    if document_store is not None:
        from dataclasses import replace

        wiring_context = replace(wiring_context, document_store=document_store)
    if key_value_cache is not None:
        from dataclasses import replace

        wiring_context = replace(wiring_context, key_value_cache=key_value_cache)
    codecraft_wiring = wire_application_codecraft(
        env, producer_adapter=resolve_environment_llm_adapter(env)
    )
    wiring_context = apply_codecraft_to_wiring_context(wiring_context, codecraft_wiring)
    from dataclasses import replace

    wiring_context = replace(
        wiring_context,
        extras={
            **wiring_context.extras,
            "effective_environment_profile": env,
        },
    )
    if resolved_integration is not None:
        wiring_context = replace(
            wiring_context,
            extras={
                **wiring_context.extras,
                "integration_profile": resolved_integration,
            },
        )

    memory_wiring = resolve_memory_platform_wiring(
        env, integration_profile=resolved_integration
    )
    assert_strict_memory_bootstrap_acceptable(env, memory_wiring)
    from intergrax.applications._shared.memory_wiring import (
        build_session_manager_from_environment,
    )

    if tenant_id is None:
        from intergrax.runtime.nexus.session.session_manager import SessionManager

        session_manager = SessionManager(memory_wiring.session_storage)
        user_profile_manager = None
    else:
        session_manager = build_session_manager_from_environment(
            env,
            tenant_id=tenant_id,
            integration_profile=resolved_integration,
            memory_wiring=memory_wiring,
            rag_stack=rag_stack,
        )
        user_profile_manager = build_user_profile_manager(
            memory_wiring.user_profile_store,
            env,
            tenant_id=tenant_id,
            rag_stack=rag_stack,
        )
    if user_profile_manager is not None:
        from dataclasses import replace

        wiring_context = replace(
            wiring_context,
            user_profile_manager=user_profile_manager,
            extras={**wiring_context.extras, "session_manager": session_manager},
        )
    else:
        from dataclasses import replace

        wiring_context = replace(
            wiring_context,
            extras={**wiring_context.extras, "session_manager": session_manager},
        )
    wiring_context = wire_session_storage_tool_binding(
        wiring_context, memory_wiring.session_storage
    )
    wiring_context = wire_scheduled_notification_tool_binding(wiring_context)
    wiring_context = _merge_integration_read_allowlist_roots(
        wiring_context, resolved_integration
    )

    hosted_session = None
    if wiring_context.sandbox_session is None and resolved_integration is not None:
        from intergrax.applications._shared.sandbox_host_wiring import (
            resolve_hosted_sandbox_session,
        )

        hosted_session = resolve_hosted_sandbox_session(
            resolved_integration,
            tenant_id="harness",
            task_id="bootstrap",
        )

    tool_wiring = build_application_tool_wiring(
        tool_profile,
        catalog_bootstrap=catalog_bootstrap,
        integration_profile=resolved_integration,
        wiring_context=wiring_context,
        vectorstore_manager=rag_stack.vectorstore_manager
        if rag_stack is not None
        else None,
        embedding_manager=rag_stack.embedding_manager
        if rag_stack is not None
        else host_embedding_manager,
        retriever_manager=rag_stack.retriever_manager
        if rag_stack is not None
        else None,
        reranker_manager=rag_stack.reranker_manager if rag_stack is not None else None,
        rag_profile=rag_stack.profile if rag_stack is not None else host_rag_profile,
        retrieval_service=rag_stack.retrieval_service
        if rag_stack is not None
        else None,
        rag_graph_store=rag_stack.graph_store if rag_stack is not None else None,
        toc_vectorstore_manager=rag_stack.toc_vectorstore_manager
        if rag_stack is not None
        else None,
        sandbox_session=sandbox_session or hosted_session,
        websearch_executor=websearch_executor,
        security_profile=env.security_profile,
    )
    tool_wiring = apply_application_owned_tool_registry(
        manifest,
        tool_wiring,
        application_tool_registry,
    )
    policy_bundle = wire_policy_bundle(
        env.model_copy(
            update={
                "domain_policy_fragments": {
                    **env.domain_policy_fragments,
                    **codecraft_wiring.domain_fragments,
                },
            },
        ),
        package_qualifications=platform_plugin_package_qualifications,
    )
    assert_strict_policy_bootstrap_acceptable(env, policy_bundle)
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
        boundary_event_buffer=boundary_event_buffer,
    )

    registry_snapshot = resolve_registry_snapshot(build_context)
    capability_graph = resolve_environment_capability_graph(
        manifest, env, registry_snapshot
    )

    if conformance_check:
        assert_registry_assembly_valid(registry_snapshot, env)
        assert_capability_graph_assembly_valid(
            capability_graph, registry_snapshot, manifest
        )
        EnvironmentSkillToolConsistencyCheck(fail_on_violation=False).validate_roster(
            manifest.agents,
            env,
        )
        ProfileInvariantValidator(fail_on_violation=False).validate(env)
        from intergrax.applications._shared.package_wiring import (
            assert_manifest_package_closure,
        )

        assert_manifest_package_closure(
            manifest,
            env,
            registry_snapshot,
            capability_graph=capability_graph,
        )
        assert_application_owned_tool_conformance(
            manifest,
            env,
            registry_snapshot,
            platform_tool_ids=platform_reserved_tool_ids(),
        )

    platform_plugin_evidence = build_application_platform_plugin_evidence(
        memory_report=memory_wiring.memory_store_plugin_load_report,
        context_report=context_bootstrap.load_report,
        security_report=security_bootstrap.load_report,
        tools_report=catalog_bootstrap.tool_plugin_load_report,
        skills_report=catalog_bootstrap.skill_plugin_load_report,
        policy_bundle=policy_bundle,
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
        platform_plugin_evidence=platform_plugin_evidence,
    )
