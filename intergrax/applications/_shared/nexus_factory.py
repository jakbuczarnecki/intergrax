# © Artur Czarnecki. All rights reserved.

"""Build NexusLoop from ApplicationEnvironmentProfile (Phase H-APP.3.3, ORCH-1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from intergrax.applications._shared.application_security_wiring import register_application_security_hooks
from intergrax.applications._shared.orchestration_wiring import (
    OrchestrationWiringContext,
    resolve_max_parallel_nodes,
    resolve_nexus_task_classifier,
    resolve_nexus_task_planner,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
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
) -> NexusLoop:
    """Apply orchestration and reliability profiles to ``NexusLoop`` construction."""
    orch = env.orchestration_profile
    reliability = env.reliability_profile
    retry_policy = RetryPolicy(max_retries=3)
    if orch.retry_policy_name == "strict":
        retry_policy = RetryPolicy(max_retries=1)

    wiring_context = OrchestrationWiringContext(llm_adapter=llm_adapter)
    planner = resolve_nexus_task_planner(env, wiring_context=wiring_context)
    classifier = resolve_nexus_task_classifier(registry, env)
    max_parallel_nodes = resolve_max_parallel_nodes(env)

    loop = NexusLoop(
        registry,
        classifier=classifier,
        planner=planner,
        max_parallel_nodes=max_parallel_nodes,
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
    )
    register_application_security_hooks(loop, env.security_profile)
    return loop
