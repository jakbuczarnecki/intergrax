# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Materialize Nexus context engines/managers from neutral environment profiles (EE owner zone)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.context.bootstrap import bootstrap_context_catalog
from intergrax.context.registry import ContextPluginRegistry, UnknownContextPluginError, get_context_plugin
from intergrax.contracts.context_budget import ContextBudgetPolicy
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.context.orchestrator import ContextOrchestrator
from intergrax.context.protocols import ContextEngine
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.execution.context_engine_resolver import load_context_engine
from intergrax.runtime.nexus.context.codebase_engine import CodebaseContextEngine
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.context.preset_engines import (
    ExploreChildContextEngine,
    RegulatedMinimalContextEngine,
)
from intergrax.runtime.task.task_contract import TaskExecutionOptions


def _resolve_context_plugin_registry_from_environment(
    env: ApplicationEnvironmentProfile,
) -> ContextPluginRegistry:
    bootstrap_context_catalog()
    plugin_ids = env.context_profile.context_plugin_ids or ["intergrax.builtin"]
    for plugin_id in plugin_ids:
        try:
            get_context_plugin(plugin_id)
        except UnknownContextPluginError:
            if env.execution_mode != ExecutionMode.STRICT:
                raise
    from intergrax.context.bootstrap import materialize_context_plugin_registry

    return materialize_context_plugin_registry(plugin_ids)


def _resolve_context_budget_policy(
    env: ApplicationEnvironmentProfile,
    *,
    llm_adapter: LLMAdapter | None = None,
) -> ContextBudgetPolicy:
    budget = env.context_profile.budget_policy
    if budget is not None:
        return budget
    if llm_adapter is not None:
        return ContextBudgetPolicy.from_adapter(llm_adapter)
    assembly = env.context_profile.assembly_options
    return ContextBudgetPolicy(max_chars=max(assembly.max_prior_chars, 4000))


def resolve_context_engine_from_environment(
    env: ApplicationEnvironmentProfile,
) -> ContextEngine:
    """Resolve context engine for the environment preset (CE-7.4, CE-8.2)."""
    registry = _resolve_context_plugin_registry_from_environment(env)
    engine_ref = env.context_profile.engine_ref
    preset = env.context_profile.engine_preset
    if preset == "custom" and engine_ref:
        return load_context_engine(engine_ref, registry=registry)
    if preset == "codebase":
        return CodebaseContextEngine(registry=registry)
    if preset == "regulated_minimal":
        return RegulatedMinimalContextEngine(registry=registry)
    if preset == "explore_child":
        return ExploreChildContextEngine(registry=registry)
    return DefaultNexusContextEngine(engine_id=preset, registry=registry)


def resolve_context_orchestrator_from_environment(
    env: ApplicationEnvironmentProfile,
    engine: ContextEngine,
) -> ContextOrchestrator | None:
    """Return bounded orchestrator for codebase preset only (CE-8.2)."""
    if env.context_profile.engine_preset != "codebase":
        return None
    return ContextOrchestrator(engine)


def resolve_context_engine_for_graph_node(
    env: ApplicationEnvironmentProfile,
    *,
    has_delegation: bool,
) -> ContextEngine:
    """Delegation children use ``explore_child`` preset automatically (CE-8.3)."""
    if has_delegation:
        registry = _resolve_context_plugin_registry_from_environment(env)
        return ExploreChildContextEngine(registry=registry)
    return resolve_context_engine_from_environment(env)


def resolve_context_manager_from_environment(
    env: ApplicationEnvironmentProfile,
    *,
    event_bus: RuntimeEventBus | None = None,
    llm_adapter: LLMAdapter | None = None,
    context_engine: ContextEngine | None = None,
) -> ContextManager:
    """Build ``ContextManager`` with environment assembly and budget policies."""
    assembly = env.context_profile.assembly_options
    engine = context_engine or resolve_context_engine_from_environment(env)
    orchestrator = resolve_context_orchestrator_from_environment(env, engine)
    return ContextManager(
        max_prior_chars=assembly.max_prior_chars,
        default_policy=assembly,
        budget_policy=_resolve_context_budget_policy(env, llm_adapter=llm_adapter),
        event_bus=event_bus,
        context_engine=engine,
        context_orchestrator=orchestrator,
        llm_adapter=llm_adapter,
    )


def merge_task_context_options_from_environment(
    options: TaskExecutionOptions,
    env: ApplicationEnvironmentProfile,
) -> TaskExecutionOptions:
    assembly = env.context_profile.assembly_options
    return options.model_copy(update={"context": assembly})


def default_task_execution_options_for_environment(
    env: ApplicationEnvironmentProfile,
) -> TaskExecutionOptions:
    return TaskExecutionOptions(
        context=env.context_profile.assembly_options,
    )


__all__ = [
    "default_task_execution_options_for_environment",
    "merge_task_context_options_from_environment",
    "resolve_context_engine_for_graph_node",
    "resolve_context_engine_from_environment",
    "resolve_context_manager_from_environment",
    "resolve_context_orchestrator_from_environment",
]
