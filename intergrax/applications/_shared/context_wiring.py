# © Artur Czarnecki. All rights reserved.

"""Tier-3 context engineering wiring (Phase CTX-2)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.task.task_contract import TaskExecutionOptions


def resolve_context_budget_policy(env: ApplicationEnvironmentProfile) -> ContextBudgetPolicy:
    """Resolve effective context budget from environment profile."""
    budget = env.context_profile.budget_policy
    if budget is not None:
        return budget
    assembly = env.context_profile.assembly_options
    return ContextBudgetPolicy(max_chars=max(assembly.max_prior_chars, 4000))


def resolve_context_manager_from_environment(
    env: ApplicationEnvironmentProfile,
    *,
    event_bus: RuntimeEventBus | None = None,
) -> ContextManager:
    """Build ``ContextManager`` with environment assembly and budget policies."""
    assembly = env.context_profile.assembly_options
    return ContextManager(
        max_prior_chars=assembly.max_prior_chars,
        default_policy=assembly,
        budget_policy=resolve_context_budget_policy(env),
        event_bus=event_bus,
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
