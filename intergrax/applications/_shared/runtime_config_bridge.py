# © Artur Czarnecki. All rights reserved.

"""Bridge Tier-3 environment into Nexus RuntimeConfig (Phase H-APP.1.5, MEM-1)."""

from __future__ import annotations

from intergrax.agents.reference_harness import LabHarnessContext
from intergrax.applications._shared.catalog_runtime_bridge import (
    apply_catalog_profiles_from_build_context,
    apply_catalog_profiles_from_environment,
)
from intergrax.applications._shared.integration_runtime_bridge import (
    apply_integration_profiles_from_build_context,
    apply_integration_profiles_from_environment,
)
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications._shared.rag_runtime_bridge import apply_rag_for_environment
from intergrax.applications._shared.context_runtime_bridge import apply_context_profiles_from_environment
from intergrax.applications._shared.memory_runtime_bridge import apply_memory_profile_to_runtime_config
from intergrax.applications._shared.observability_runtime_bridge import (
    apply_observability_profiles_from_environment,
)
from intergrax.applications._shared.reliability_runtime_bridge import (
    apply_reliability_profiles_from_environment,
)
from intergrax.applications._shared.cost_runtime_bridge import (
    apply_cost_profiles_from_environment,
)
from intergrax.applications._shared.adaptive_runtime_bridge import apply_adaptive_profiles_from_environment
from intergrax.applications._shared.adaptive_wiring import ApplicationAdaptiveWiring, wire_adaptive_profile
from intergrax.applications._shared.critic_runtime_bridge import (
    apply_critic_profiles_from_environment,
)
from intergrax.applications._shared.evaluation_runtime_bridge import (
    apply_evaluation_profiles_from_environment,
)
from intergrax.applications._shared.evaluation_wiring import wire_application_evaluation
from intergrax.applications._shared.reliability_wiring import wire_application_reliability
from intergrax.applications._shared.security_runtime_bridge import (
    apply_security_profiles_from_environment,
)
from intergrax.applications._shared.prompt_runtime_bridge import apply_prompt_profiles_from_environment
from intergrax.applications._shared.prompt_wiring import resolve_prompt_registry
from intergrax.applications._shared.memory_wiring import (
    build_session_manager_from_environment,
    resolve_memory_platform_wiring,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import runtime_policies_for_execution_mode
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.tools.scope_policy import ToolScopePolicy


def apply_policy_bundle_to_runtime_config(
    config: RuntimeConfig,
    bundle: RuntimePolicyBundle | None,
) -> RuntimeConfig:
    """Attach composed policy bundle; fill Nexus budget/plan-loop slots when unset."""
    if bundle is None:
        return config
    config.policy_bundle = bundle
    if bundle.budget is not None and config.budget_policy is None:
        config.budget_policy = bundle.budget
    if bundle.plan_loop is not None and config.plan_loop_policy is None:
        config.plan_loop_policy = bundle.plan_loop
    if bundle.tool_access is not None and config.tool_scope_policy is None:
        if isinstance(bundle.tool_access, ToolScopePolicy):
            config.tool_scope_policy = bundle.tool_access
    return config


def materialize_runtime_config(
    request: RuntimeRequest,
    harness_ctx: LabHarnessContext | ApplicationBuildContext,
    env: ApplicationEnvironmentProfile,
    *,
    llm_adapter: LLMAdapter | None = None,
    pipeline: RuntimePipeline | None = None,
) -> RuntimeConfig:
    """
    Map ``ApplicationEnvironmentProfile`` → ``RuntimeConfig`` (H-APP.1.5, MEM-1).
    """
    ctx_profile = env.context_profile
    trace_path: str | None = None
    strict = False
    policy_bundle: RuntimePolicyBundle | None = None
    modality_profile = env.modality_profile
    tool_wiring_context = None
    integration_profile = env.integration_profile

    if isinstance(harness_ctx, LabHarnessContext):
        strict = harness_ctx.strict_harness
        if harness_ctx.trace_db_path is not None:
            trace_path = str(harness_ctx.trace_db_path)
        policy_bundle = harness_ctx.policy_bundle
        if harness_ctx.modality_profile is not None:
            modality_profile = harness_ctx.modality_profile
        tool_wiring_context = harness_ctx.tool_wiring_context
    elif isinstance(harness_ctx, ApplicationBuildContext):
        strict = harness_ctx.strict_harness
        if harness_ctx.trace_db_path is not None:
            trace_path = str(harness_ctx.trace_db_path)
        policy_bundle = harness_ctx.policy_bundle
        tool_wiring_context = harness_ctx.tool_wiring_context
        if harness_ctx.integration_profile is not None:
            integration_profile = harness_ctx.integration_profile

    resolved_llm = resolve_llm_adapter(env, agent_override=llm_adapter)

    config = RuntimeConfig(
        llm_adapter=resolved_llm,
        enable_rag=ctx_profile.enable_rag,
        enable_websearch=ctx_profile.enable_websearch,
        production_mode=strict or env.execution_mode.value == "strict",
        tenant_id=request.tenant_id,
        trace_db_path=trace_path,
        security_profile=env.security_profile,
        modality_profile=modality_profile,
        tool_wiring_context=tool_wiring_context,
        runtime_policies=runtime_policies_for_execution_mode(env.execution_mode),
        integration_profile=integration_profile,
    )
    if pipeline is not None:
        config.pipeline = pipeline
    apply_memory_profile_to_runtime_config(config, env.memory_profile)
    apply_prompt_profiles_from_environment(config, env)
    apply_observability_profiles_from_environment(config, env)
    apply_security_profiles_from_environment(config, env)
    reliability_wiring = wire_application_reliability(env)
    apply_reliability_profiles_from_environment(
        config,
        env,
        idempotency_store=reliability_wiring.idempotency_store,
    )
    apply_context_profiles_from_environment(config, env)
    apply_cost_profiles_from_environment(config, env)
    evaluation_wiring = wire_application_evaluation(env)
    adaptive_wiring = wire_adaptive_profile(
        env,
        evaluation_governance_bridge=evaluation_wiring.governance_bridge,
        tenant_id=request.tenant_id,
    )
    apply_evaluation_profiles_from_environment(
        config,
        env,
        registry=evaluation_wiring.registry,
    )
    apply_critic_profiles_from_environment(config, env)
    apply_adaptive_profiles_from_environment(
        config,
        env,
        wiring=adaptive_wiring,
        tenant_id=request.tenant_id,
        task_class=str(request.metadata.get("task_class", "")),
        routing_key=str(request.metadata.get("run_id", request.session_id)),
    )
    apply_integration_profiles_from_environment(config, env)
    apply_catalog_profiles_from_environment(config, env)
    if isinstance(harness_ctx, ApplicationBuildContext):
        apply_integration_profiles_from_build_context(config, harness_ctx)
        apply_catalog_profiles_from_build_context(config, harness_ctx)
    rag_wiring_context = tool_wiring_context
    if isinstance(harness_ctx, ApplicationBuildContext) and harness_ctx.tool_wiring_context is not None:
        rag_wiring_context = harness_ctx.tool_wiring_context
    apply_rag_for_environment(config, env, tool_wiring_context=rag_wiring_context)
    return apply_policy_bundle_to_runtime_config(config, policy_bundle)


def build_runtime_context_from_environment(
    request: RuntimeRequest,
    harness_ctx: LabHarnessContext | ApplicationBuildContext,
    env: ApplicationEnvironmentProfile,
    *,
    llm_adapter: LLMAdapter | None = None,
    pipeline: RuntimePipeline | None = None,
) -> RuntimeContext:
    """Build ``RuntimeContext`` from environment profile with resolved memory backends."""
    config = materialize_runtime_config(
        request,
        harness_ctx,
        env,
        llm_adapter=llm_adapter,
        pipeline=pipeline,
    )
    integration_profile = config.integration_profile or env.integration_profile
    memory_wiring = resolve_memory_platform_wiring(
        env,
        integration_profile=integration_profile,
    )
    session_manager = build_session_manager_from_environment(
        env,
        integration_profile=integration_profile,
        memory_wiring=memory_wiring,
    )
    prompt_registry = resolve_prompt_registry(env.prompt_profile)
    return RuntimeContext.build(
        config=config,
        session_manager=session_manager,
        prompt_registry=prompt_registry,
    )
