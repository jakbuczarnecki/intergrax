# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Materialize Nexus context engines/managers from neutral environment profiles (EE owner zone)."""

from __future__ import annotations

from intergrax.context.bootstrap import bootstrap_context_catalog
from intergrax.context.orchestrator import ContextOrchestrator
from intergrax.context.protocols import ContextEngine
from intergrax.context.registry import ContextPluginRegistry, UnknownContextPluginError, get_context_plugin
from intergrax.contracts.context_budget import ContextBudgetPolicy
from intergrax.contracts.host_orchestration_environment import HostContextCompositionSpec
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
    spec: HostContextCompositionSpec,
) -> ContextPluginRegistry:
    bootstrap_context_catalog()
    plugin_ids = list(spec.context_plugin_ids) or ["intergrax.builtin"]
    for plugin_id in plugin_ids:
        try:
            get_context_plugin(plugin_id)
        except UnknownContextPluginError:
            if not spec.strict:
                raise
    from intergrax.context.bootstrap import materialize_context_plugin_registry

    return materialize_context_plugin_registry(plugin_ids)


def _resolve_context_budget_policy(
    spec: HostContextCompositionSpec,
    *,
    llm_adapter: LLMAdapter | None = None,
) -> ContextBudgetPolicy:
    budget = spec.budget_policy
    if budget is not None:
        return budget
    if llm_adapter is not None:
        return ContextBudgetPolicy.from_adapter(llm_adapter)
    assembly = spec.assembly_options
    return ContextBudgetPolicy(max_chars=max(assembly.max_prior_chars, 4000))


def resolve_context_engine_from_environment(
    spec: HostContextCompositionSpec,
) -> ContextEngine:
    """Resolve context engine for the environment preset (CE-7.4, CE-8.2)."""
    registry = _resolve_context_plugin_registry_from_environment(spec)
    engine_ref = spec.engine_ref
    preset = spec.engine_preset
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
    spec: HostContextCompositionSpec,
    engine: ContextEngine,
) -> ContextOrchestrator | None:
    """Return bounded orchestrator for codebase preset only (CE-8.2)."""
    if spec.engine_preset != "codebase":
        return None
    return ContextOrchestrator(engine)


def resolve_context_engine_for_graph_node(
    spec: HostContextCompositionSpec,
    *,
    has_delegation: bool,
) -> ContextEngine:
    """Delegation children use ``explore_child`` preset automatically (CE-8.3)."""
    if has_delegation:
        registry = _resolve_context_plugin_registry_from_environment(spec)
        return ExploreChildContextEngine(registry=registry)
    return resolve_context_engine_from_environment(spec)


def resolve_context_manager_from_environment(
    spec: HostContextCompositionSpec,
    *,
    event_bus: RuntimeEventBus | None = None,
    llm_adapter: LLMAdapter | None = None,
    context_engine: ContextEngine | None = None,
) -> ContextManager:
    """Build ``ContextManager`` with environment assembly and budget policies."""
    assembly = spec.assembly_options
    engine = context_engine or resolve_context_engine_from_environment(spec)
    orchestrator = resolve_context_orchestrator_from_environment(spec, engine)
    return ContextManager(
        max_prior_chars=assembly.max_prior_chars,
        default_policy=assembly,
        budget_policy=_resolve_context_budget_policy(spec, llm_adapter=llm_adapter),
        event_bus=event_bus,
        context_engine=engine,
        context_orchestrator=orchestrator,
        llm_adapter=llm_adapter,
    )


def merge_task_context_options_from_environment(
    options: TaskExecutionOptions,
    spec: HostContextCompositionSpec,
) -> TaskExecutionOptions:
    return options.model_copy(update={"context": spec.assembly_options})


def default_task_execution_options_for_environment(
    spec: HostContextCompositionSpec,
) -> TaskExecutionOptions:
    return TaskExecutionOptions(
        context=spec.assembly_options,
    )


__all__ = [
    "ContextManager",
    "default_task_execution_options_for_environment",
    "merge_task_context_options_from_environment",
    "resolve_context_engine_for_graph_node",
    "resolve_context_engine_from_environment",
    "resolve_context_manager_from_environment",
    "resolve_context_orchestrator_from_environment",
]
