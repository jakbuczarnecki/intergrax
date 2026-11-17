# © Artur Czarnecki. All rights reserved.

"""Tier-3 resolution of neutral host-orchestration values from ``ApplicationEnvironmentProfile``.

Tier-3 reads the profile by explicit named fields and hands frozen neutral values to the
Execution Engine (Tier-1), which never imports the profile models.
"""

from __future__ import annotations

from intergrax.applications._shared.delegation_budget_wiring import (
    resolve_delegation_budget_policy,
)
from intergrax.applications._shared.orchestration_wiring import (
    OrchestrationWiringContext,
    OrchestrationWiringError,
    resolve_orchestration_runtime_settings,
)
from intergrax.applications._shared.reasoning_wiring import resolve_planner_model_id
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec
from intergrax.contracts.execution_mode import ExecutionMode
from intergrax.contracts.host_orchestration_environment import (
    HostContextCompositionSpec,
    HostOrchestrationEnvironment,
    HostOrchestrationPlanningSpec,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.execution.host_orchestration_planner_classifier_wiring import (
    NexusTaskClassifierProtocol,
    NexusTaskPlannerProtocol,
)
from intergrax.runtime.execution.host_orchestration_planner_classifier_wiring import (
    OrchestrationWiringError as _ExecutionOrchestrationWiringError,
)
from intergrax.runtime.execution.host_orchestration_planner_classifier_wiring import (
    resolve_nexus_task_classifier as _resolve_nexus_task_classifier,
)
from intergrax.runtime.execution.host_orchestration_planner_classifier_wiring import (
    resolve_nexus_task_planner as _resolve_nexus_task_planner,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry


class LiveEnvironmentGraphTopologySource:
    """Live read of ``env.graph_spec`` so the planner sees a replaced topology at plan time."""

    def __init__(self, environment: ApplicationEnvironmentProfile) -> None:
        self._environment = environment

    def current_graph_topology(self) -> ApplicationGraphSpec | None:
        return self._environment.graph_spec


def host_context_composition_spec(
    env: ApplicationEnvironmentProfile,
) -> HostContextCompositionSpec:
    """Resolve context engine and manager inputs from the environment context profile."""
    context = env.context_profile
    return HostContextCompositionSpec(
        context_plugin_ids=tuple(context.context_plugin_ids),
        assembly_options=context.assembly_options,
        budget_policy=context.budget_policy,
        engine_preset=context.engine_preset,
        engine_ref=context.engine_ref,
        strict=env.execution_mode == ExecutionMode.STRICT,
    )


def host_orchestration_planning_spec(
    env: ApplicationEnvironmentProfile,
    *,
    llm_adapter: LLMAdapter | None = None,
    planner_llm_adapter: LLMAdapter | None = None,
) -> HostOrchestrationPlanningSpec:
    """Resolve planner and classifier inputs from the orchestration and reasoning profiles."""
    orchestration = env.orchestration_profile
    reasoning = env.reasoning_profile
    settings = resolve_orchestration_runtime_settings(env)
    return HostOrchestrationPlanningSpec(
        planner_kind=orchestration.planner_kind,
        classifier_kind=orchestration.classifier_kind,
        multi_agent_order=settings.multi_agent_order,
        coordination_pattern=orchestration.coordination_pattern,
        intent_routes=tuple(orchestration.intent_routes),
        planner_prompt_id=reasoning.planner_prompt_id,
        classifier_prompt_id=reasoning.classifier_prompt_id,
        planner_parse_retries=reasoning.planner_parse_retries,
        llm_adapter=llm_adapter,
        planner_llm_adapter=planner_llm_adapter,
        graph_topology_source=LiveEnvironmentGraphTopologySource(env),
        delegation_budget=resolve_delegation_budget_policy(env),
    )


def host_orchestration_environment(
    env: ApplicationEnvironmentProfile,
) -> HostOrchestrationEnvironment:
    """Resolve execution-engine inputs from the orchestration, reliability and reasoning profiles."""
    orchestration = env.orchestration_profile
    reliability = env.reliability_profile
    settings = resolve_orchestration_runtime_settings(env)
    return HostOrchestrationEnvironment(
        execution_mode=env.execution_mode,
        retry_policy_name=orchestration.retry_policy_name,
        long_running_scheduler_enabled=reliability.long_running_scheduler_enabled,
        attempt_lifecycle_persistence_provider=(reliability.attempt_lifecycle_persistence_provider),
        execution_lineage_persistence_provider=(reliability.execution_lineage_persistence_provider),
        execution_terminal_persistence_provider=(reliability.execution_terminal_persistence_provider),
        max_parallel_nodes=settings.max_parallel_nodes,
        max_inflight_nodes=settings.max_inflight_nodes,
        max_delegation_depth=settings.max_delegation_depth,
        max_run_retries=settings.max_run_retries,
        merge_strategy=settings.merge_strategy,
        allow_dynamic_replan=settings.allow_dynamic_replan,
        emit_coordination_advisory=orchestration.emit_coordination_advisory,
        denied_planner_model_ids=tuple(env.reasoning_profile.denied_planner_model_ids),
        planner_model_id=resolve_planner_model_id(env),
    )


def _planning_spec_from_context(
    env: ApplicationEnvironmentProfile,
    wiring_context: OrchestrationWiringContext | None,
) -> HostOrchestrationPlanningSpec:
    context = wiring_context or OrchestrationWiringContext()
    return host_orchestration_planning_spec(
        env,
        llm_adapter=context.llm_adapter,
        planner_llm_adapter=context.planner_llm_adapter,
    )


def resolve_nexus_task_planner_from_profile(
    env: ApplicationEnvironmentProfile,
    *,
    wiring_context: OrchestrationWiringContext | None = None,
) -> NexusTaskPlannerProtocol:
    """Build the Tier-1 planner from the profile (errors surface as Tier-3 wiring errors)."""
    try:
        return _resolve_nexus_task_planner(_planning_spec_from_context(env, wiring_context))
    except _ExecutionOrchestrationWiringError as exc:
        raise OrchestrationWiringError(str(exc)) from exc


def resolve_nexus_task_classifier_from_profile(
    registry: AgentRegistry,
    env: ApplicationEnvironmentProfile,
    *,
    wiring_context: OrchestrationWiringContext | None = None,
) -> NexusTaskClassifierProtocol:
    """Build the Tier-1 classifier from the profile (errors surface as Tier-3 wiring errors)."""
    try:
        return _resolve_nexus_task_classifier(
            registry,
            _planning_spec_from_context(env, wiring_context),
        )
    except _ExecutionOrchestrationWiringError as exc:
        raise OrchestrationWiringError(str(exc)) from exc


__all__ = [
    "LiveEnvironmentGraphTopologySource",
    "host_context_composition_spec",
    "host_orchestration_environment",
    "host_orchestration_planning_spec",
    "resolve_nexus_task_classifier_from_profile",
    "resolve_nexus_task_planner_from_profile",
]
