# © Artur Czarnecki. All rights reserved.

"""Tier-3 context engineering wiring (Phase CTX-2, CE-2.4)."""

from __future__ import annotations

import logging

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.context.bootstrap import bootstrap_context_catalog, materialize_context_plugin_registry
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.context.registry import ContextPluginRegistry, UnknownContextPluginError
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.task.task_contract import TaskExecutionOptions

logger = logging.getLogger(__name__)


def _is_production_environment(env: ApplicationEnvironmentProfile) -> bool:
    """Lab / dev profiles fail closed on unknown plugin ids; prod hosts warn."""
    from intergrax.applications.contracts.execution_mode import ExecutionMode

    return env.execution_mode == ExecutionMode.STRICT


def bootstrap_application_context_catalog(
    *,
    discover_entry_points: bool | None = None,
) -> None:
    """Register shipped context catalog (and optional entry-point plugins)."""
    discover = discover_plugins_enabled() if discover_entry_points is None else discover_entry_points
    bootstrap_context_catalog(discover_entry_points=discover)


def validate_context_plugin_ids(
    env: ApplicationEnvironmentProfile,
    *,
    production_mode: bool = True,
) -> list[str]:
    """
    Validate ``ContextProfile.context_plugin_ids`` against the catalog.

    Lab hosts (``production_mode=False``) fail closed; production warns.
    """
    plugin_ids = list(env.context_profile.context_plugin_ids)
    if not plugin_ids:
        return []

    bootstrap_application_context_catalog()
    unknown: list[str] = []
    from intergrax.context.registry import get_context_plugin

    for plugin_id in plugin_ids:
        try:
            get_context_plugin(plugin_id)
        except UnknownContextPluginError:
            unknown.append(plugin_id)

    if not unknown:
        return []

    message = f"Unknown context plugin id(s): {', '.join(sorted(unknown))}"
    if production_mode:
        logger.warning("%s", message)
        return unknown
    raise ValueError(message)


def resolve_context_plugin_registry_from_environment(
    env: ApplicationEnvironmentProfile,
) -> ContextPluginRegistry:
    """Materialize enabled context plugins for the environment profile."""
    bootstrap_application_context_catalog()
    validate_context_plugin_ids(env, production_mode=_is_production_environment(env))
    plugin_ids = env.context_profile.context_plugin_ids or ["intergrax.builtin"]
    return materialize_context_plugin_registry(plugin_ids)


def resolve_context_budget_policy(env: ApplicationEnvironmentProfile) -> ContextBudgetPolicy:
    """Resolve effective context budget from environment profile."""
    budget = env.context_profile.budget_policy
    if budget is not None:
        return budget
    assembly = env.context_profile.assembly_options
    return ContextBudgetPolicy(max_chars=max(assembly.max_prior_chars, 4000))


def resolve_context_engine_from_environment(
    env: ApplicationEnvironmentProfile,
) -> DefaultNexusContextEngine:
    """Resolve context engine for the environment preset (CE-7.4, CE-8.2)."""
    registry = resolve_context_plugin_registry_from_environment(env)
    engine_ref = env.context_profile.engine_ref
    preset = env.context_profile.engine_preset
    if preset == "custom" and engine_ref:
        raise ValueError(
            f"Custom context engine_ref is not wired yet: {engine_ref!r} (use preset=default)"
        )
    if preset == "codebase":
        from intergrax.runtime.nexus.context.codebase_engine import CodebaseContextEngine

        return CodebaseContextEngine(registry=registry)
    return DefaultNexusContextEngine(engine_id=preset, registry=registry)


def resolve_context_orchestrator_from_environment(
    env: ApplicationEnvironmentProfile,
    engine: DefaultNexusContextEngine,
):
    """Return bounded orchestrator for codebase preset only (CE-8.2)."""
    if env.context_profile.engine_preset != "codebase":
        return None
    from intergrax.context.orchestrator import ContextOrchestrator

    return ContextOrchestrator(engine)


def resolve_context_engine_for_graph_node(
    env: ApplicationEnvironmentProfile,
    *,
    has_delegation: bool,
) -> DefaultNexusContextEngine:
    """Delegation children use ``explore_child`` preset automatically (CE-8.3)."""
    if has_delegation:
        registry = resolve_context_plugin_registry_from_environment(env)
        return DefaultNexusContextEngine(engine_id="explore_child", registry=registry)
    return resolve_context_engine_from_environment(env)


def resolve_context_manager_from_environment(
    env: ApplicationEnvironmentProfile,
    *,
    event_bus: RuntimeEventBus | None = None,
    llm_adapter: object | None = None,
    context_engine: DefaultNexusContextEngine | None = None,
) -> ContextManager:
    """Build ``ContextManager`` with environment assembly and budget policies."""
    assembly = env.context_profile.assembly_options
    engine = context_engine or resolve_context_engine_from_environment(env)
    return ContextManager(
        max_prior_chars=assembly.max_prior_chars,
        default_policy=assembly,
        budget_policy=resolve_context_budget_policy(env),
        event_bus=event_bus,
        context_engine=engine,
        llm_adapter=llm_adapter,  # type: ignore[arg-type]
    )


def merge_task_context_options_from_environment(
    options: TaskExecutionOptions,
    env: ApplicationEnvironmentProfile,
) -> TaskExecutionOptions:
    """
    Overlay ``ContextProfile.assembly_options`` on task intake options.

    Preserves non-context fields (governance, isolation, long_running).
    """
    assembly = env.context_profile.assembly_options
    return options.model_copy(update={"context": assembly})


def default_task_execution_options_for_environment(
    env: ApplicationEnvironmentProfile,
) -> TaskExecutionOptions:
    """Baseline task intake options derived from environment context profile."""
    return TaskExecutionOptions(
        context=env.context_profile.assembly_options,
    )
