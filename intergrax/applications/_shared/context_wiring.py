# © Artur Czarnecki. All rights reserved.

"""Tier-3 context engineering wiring (Phase CTX-2, CE-2.4) — neutral profile/bootstrap only."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.context.bootstrap import (
    ContextCatalogBootstrapResult,
    bootstrap_context_catalog,
    materialize_context_plugin_registry,
)
from intergrax.context.registry import ContextPluginRegistry, UnknownContextPluginError
from intergrax.context.protocols import ContextEngine
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.contracts.context_budget import ContextBudgetPolicy
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.execution.host_runtime_config import RuntimeConfig
from intergrax.runtime.execution.application_environment_context_composition import (
    default_task_execution_options_for_environment,
    merge_task_context_options_from_environment,
    resolve_context_engine_for_graph_node,
    resolve_context_engine_from_environment,
    resolve_context_manager_from_environment,
    resolve_context_orchestrator_from_environment,
)
from intergrax.runtime.task.task_contract import TaskExecutionOptions

logger = logging.getLogger(__name__)

__all__ = [
    "ContextAssemblyError",
    "apply_context_engine_to_runtime_config",
    "assert_strict_context_bootstrap_acceptable",
    "bootstrap_application_context_catalog",
    "context_plugin_bootstrap_errors",
    "default_task_execution_options_for_environment",
    "merge_task_context_options_from_environment",
    "resolve_context_budget_policy",
    "resolve_context_engine_for_graph_node",
    "resolve_context_engine_from_environment",
    "resolve_context_manager_from_environment",
    "resolve_context_orchestrator_from_environment",
    "resolve_context_plugin_registry_from_environment",
    "validate_context_plugin_ids",
]


class ContextAssemblyError(ValueError):
    """Raised when context assembly validation fails."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def context_plugin_bootstrap_errors(report: DomainPluginLoadReport) -> tuple[str, ...]:
    errors: list[str] = []
    for item in report.failed:
        errors.append(f"context plugin load failed: {item.spec.name}: {item.error}")
    for item in report.rejected:
        if item.fail_closed:
            errors.append(
                "context plugin admission rejected: "
                f"{item.spec.name}: {item.reason_code.value}",
            )
    if not errors:
        errors.append("context plugin bootstrap admission is not acceptable")
    return tuple(errors)


def assert_strict_context_bootstrap_acceptable(
    env: ApplicationEnvironmentProfile,
    context_bootstrap: ContextCatalogBootstrapResult,
) -> None:
    if env.execution_mode is not ExecutionMode.STRICT:
        return
    if context_bootstrap.load_report.critical_bootstrap_acceptable:
        return
    raise ContextAssemblyError(
        context_plugin_bootstrap_errors(context_bootstrap.load_report),
    )


def _is_production_environment(env: ApplicationEnvironmentProfile) -> bool:
    return env.execution_mode == ExecutionMode.STRICT


def bootstrap_application_context_catalog(
    *,
    discover_entry_points: bool | None = None,
) -> ContextCatalogBootstrapResult:
    """Register shipped context catalog (and optional entry-point plugins)."""
    discover = discover_plugins_enabled() if discover_entry_points is None else discover_entry_points
    return bootstrap_context_catalog(discover_entry_points=discover)


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


def resolve_context_budget_policy(
    env: ApplicationEnvironmentProfile,
    *,
    llm_adapter: LLMAdapter | None = None,
) -> ContextBudgetPolicy:
    """Resolve effective context budget from environment profile."""
    budget = env.context_profile.budget_policy
    if budget is not None:
        return budget
    if llm_adapter is not None:
        return ContextBudgetPolicy.from_adapter(llm_adapter)
    assembly = env.context_profile.assembly_options
    return ContextBudgetPolicy(max_chars=max(assembly.max_prior_chars, 4000))


def apply_context_engine_to_runtime_config(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
    *,
    context_engine: ContextEngine | None = None,
) -> RuntimeConfig:
    """Resolve and inject ``ContextEngine`` at host composition (MEM-XINT-4-R2)."""
    if config.context_engine is not None:
        return config
    config.context_engine = context_engine or resolve_context_engine_from_environment(env)
    return config
