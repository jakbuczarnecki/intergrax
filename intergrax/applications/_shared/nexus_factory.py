# © Artur Czarnecki. All rights reserved.

"""Build NexusLoop from ApplicationEnvironmentProfile (Phase H-APP.3.3, ORCH-1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.applications._shared.critic_wiring import (
    ApplicationCriticWiring,
    apply_application_critic_wiring,
)
from intergrax.applications._shared.guardrail_wiring import (
    ApplicationGuardrailWiring,
    apply_application_guardrail_wiring,
    wire_application_guardrail,
)
from intergrax.applications._shared.security_wiring import (
    ApplicationSecurityWiring,
    apply_application_security_wiring,
    wire_application_security,
)
from intergrax.applications._shared.context_wiring import resolve_context_manager_from_environment
from intergrax.applications._shared.orchestration_wiring import (
    OrchestrationWiringContext,
    resolve_nexus_task_classifier,
    resolve_nexus_task_planner,
    resolve_orchestration_runtime_settings,
)
from intergrax.applications._shared.adaptive_wiring import ApplicationAdaptiveWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager


def build_nexus_loop_from_environment(
    registry: AgentRegistry,
    *,
    env: ApplicationEnvironmentProfile,
    trace_store: RunTraceWriter | None = None,
    checkpoint_store: SQLiteTaskCheckpointStore | None = None,
    notification_adapter: NotificationAdapter | None = None,
    runtime_events_db_path: Path | None = None,
    task_memory_store: Any | None = None,
    task_memory_db_path: Path | None = None,
    shadow_manager: ShadowWorkspaceManager | None = None,
    sandbox_manager: SandboxSessionManager | None = None,
    llm_adapter: LLMAdapter | None = None,
    runtime_event_bus: RuntimeEventBus | None = None,
    context_manager: ContextManager | None = None,
    security_wiring: ApplicationSecurityWiring | None = None,
    guardrail_wiring: ApplicationGuardrailWiring | None = None,
    critic_wiring: ApplicationCriticWiring | None = None,
    adaptive_wiring: ApplicationAdaptiveWiring | None = None,
    run_budget: RunBudget | None = None,
) -> NexusLoop:
    """Apply orchestration and reliability profiles to ``NexusLoop`` construction."""
    orch = env.orchestration_profile
    reliability = env.reliability_profile
    retry_policy = RetryPolicy(max_retries=3)
    if orch.retry_policy_name == "strict":
        retry_policy = RetryPolicy(max_retries=1)

    wiring_context = OrchestrationWiringContext(llm_adapter=llm_adapter)
    planner = resolve_nexus_task_planner(env, wiring_context=wiring_context)
    classifier = resolve_nexus_task_classifier(registry, env, wiring_context=wiring_context)
    runtime_settings = resolve_orchestration_runtime_settings(env)
    resolved_context_manager = context_manager or resolve_context_manager_from_environment(
        env,
        event_bus=runtime_event_bus,
    )

    loop = NexusLoop(
        registry,
        classifier=classifier,
        planner=planner,
        max_parallel_nodes=runtime_settings.max_parallel_nodes,
        max_inflight_nodes=runtime_settings.max_inflight_nodes,
        max_delegation_depth=runtime_settings.max_delegation_depth,
        max_run_retries=runtime_settings.max_run_retries,
        merge_strategy=runtime_settings.merge_strategy,
        context_manager=resolved_context_manager,
        event_bus=runtime_event_bus,
        trace_store=trace_store,
        retry_policy=retry_policy,
        shadow_manager=shadow_manager,
        sandbox_manager=sandbox_manager,
        checkpoint_store=checkpoint_store
        if reliability.long_running_scheduler_enabled
        else None,
        notification_adapter=notification_adapter,
        runtime_events_db_path=runtime_events_db_path,
        task_memory_store=task_memory_store,
        task_memory_db_path=task_memory_db_path,
        production_mode=env.execution_mode.value == "strict",
        signal_collector=adaptive_wiring.signal_collector if adaptive_wiring else None,
        run_budget=run_budget,
        critic_graph_hooks=critic_wiring.graph_hooks if critic_wiring else None,
        emit_coordination_advisory=orch.emit_coordination_advisory,
        allow_dynamic_replan=runtime_settings.allow_dynamic_replan,
        denied_planner_model_ids=tuple(env.reasoning_profile.denied_planner_model_ids),
        planner_model_id=env.reasoning_profile.planner_llm_profile_id,
    )
    resolved_security = security_wiring or wire_application_security(env)
    apply_application_security_wiring(loop, resolved_security)
    resolved_guardrail = guardrail_wiring or wire_application_guardrail(env)
    apply_application_guardrail_wiring(loop, resolved_guardrail, env)
    if critic_wiring is not None:
        apply_application_critic_wiring(loop, critic_wiring)
    return loop
