# © Artur Czarnecki. All rights reserved.

"""Build NexusLoop from ApplicationEnvironmentProfile (Phase H-APP.3.3, ORCH-1)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from intergrax.applications._shared.llm_resolver import resolve_environment_llm_adapter
from intergrax.applications._shared.orchestration_wiring import (
    OrchestrationWiringContext,
    resolve_nexus_task_classifier,
    resolve_nexus_task_planner,
    resolve_orchestration_runtime_settings,
)
from intergrax.applications._shared.reasoning_wiring import (
    resolve_planner_llm_adapter,
    resolve_planner_model_id,
)
from intergrax.applications._shared.adaptive_wiring import ApplicationAdaptiveWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.agents.persistence.checkpoint_store import AgentCheckpointStore
from intergrax.agents.persistence.compensation_queue_store import CompensationQueueStore
from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvoker
from intergrax.runtime.execution.authority import (
    resolve_execution_authority_policy_from_runtime_config,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine
from intergrax.runtime.nexus.retry.retry_engine import RetryPolicy
from intergrax.runtime.nexus.tracing.persistence_models import RunTraceWriter
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.sandbox.manager import SandboxSessionManager
from intergrax.runtime.workspace.manager import ShadowWorkspaceManager


if TYPE_CHECKING:
    from intergrax.runtime.execution.authority.policy import ExecutionAuthorityPolicy


def build_nexus_loop_from_environment(
    registry: AgentRegistry,
    *,
    env: ApplicationEnvironmentProfile,
    trace_store: RunTraceWriter | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    agent_checkpoint_store: AgentCheckpointStore | None = None,
    compensation_queue_store: CompensationQueueStore | None = None,
    idempotency_store: IdempotencyStore | None = None,
    declarative_tool_invoker: DeclarativeToolInvoker | None = None,
    notification_adapter: NotificationAdapter | None = None,
    runtime_events_db_path: Path | None = None,
    task_memory_store: Any | None = None,
    task_memory_db_path: Path | None = None,
    shadow_manager: ShadowWorkspaceManager | None = None,
    sandbox_manager: SandboxSessionManager | None = None,
    llm_adapter: LLMAdapter | None = None,
    runtime_event_bus: RuntimeEventBus | None = None,
    context_manager: ContextManager | None = None,
    context_engine: object | None = None,
    security_wiring: ApplicationSecurityWiring | None = None,
    guardrail_wiring: ApplicationGuardrailWiring | None = None,
    critic_wiring: ApplicationCriticWiring | None = None,
    adaptive_wiring: ApplicationAdaptiveWiring | None = None,
    run_budget: RunBudget | None = None,
    validation_engine: NexusValidationEngine | None = None,
    runtime_config: RuntimeConfig | None = None,
    authority_policy: ExecutionAuthorityPolicy | None = None,
) -> NexusLoop:
    """Apply orchestration and reliability profiles to ``NexusLoop`` construction."""
    orch = env.orchestration_profile
    reliability = env.reliability_profile
    retry_policy = RetryPolicy(max_retries=3)
    if orch.retry_policy_name == "strict":
        retry_policy = RetryPolicy(max_retries=1)

    producer_llm = resolve_environment_llm_adapter(env, agent_override=llm_adapter)
    planner_llm = resolve_planner_llm_adapter(env, producer_adapter=producer_llm)
    wiring_context = OrchestrationWiringContext(
        llm_adapter=producer_llm,
        planner_llm_adapter=planner_llm,
        planner_parse_retries=env.reasoning_profile.planner_parse_retries,
    )
    planner = resolve_nexus_task_planner(env, wiring_context=wiring_context)
    classifier = resolve_nexus_task_classifier(registry, env, wiring_context=wiring_context)
    runtime_settings = resolve_orchestration_runtime_settings(env)
    resolved_authority_policy = authority_policy
    if resolved_authority_policy is None and runtime_config is not None:
        resolved_authority_policy = resolve_execution_authority_policy_from_runtime_config(
            runtime_config,
        )
    resolved_context_manager = context_manager or resolve_context_manager_from_environment(
        env,
        event_bus=runtime_event_bus,
        llm_adapter=producer_llm,
        context_engine=context_engine,  # type: ignore[arg-type]
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
        agent_checkpoint_store=agent_checkpoint_store,
        compensation_queue_store=compensation_queue_store,
        idempotency_store=idempotency_store,
        declarative_tool_invoker=declarative_tool_invoker,
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
        planner_model_id=resolve_planner_model_id(env),
        validation_engine=validation_engine,
        authority_policy=resolved_authority_policy,
    )
    resolved_security = security_wiring or wire_application_security(env)
    apply_application_security_wiring(loop, resolved_security, env=env)
    resolved_guardrail = guardrail_wiring or wire_application_guardrail(env)
    apply_application_guardrail_wiring(loop, resolved_guardrail, env)
    if critic_wiring is not None:
        apply_application_critic_wiring(loop, critic_wiring)
    return loop
