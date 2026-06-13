# © Artur Czarnecki. All rights reserved.

"""CTX-2: ContextManager and task options wiring from environment."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.context_wiring import (
    default_task_execution_options_for_environment,
    merge_task_context_options_from_environment,
    resolve_context_budget_policy,
    resolve_context_manager_from_environment,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications.contracts.environment_profile import ContextProfile
from intergrax.contracts.context_assembly import ContextSummaryTier, TaskContextAssemblyOptions
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.task.task_contract import TaskExecutionOptions
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_resolve_context_budget_policy_uses_profile_or_derives_from_assembly() -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env_with_budget = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "context_profile": ContextProfile(
                budget_policy=ContextBudgetPolicy(max_tokens_estimate=2_500),
            ),
        }
    )
    assert resolve_context_budget_policy(env_with_budget).max_tokens_estimate == 2_500

    env_default = ApplicationEnvironmentProfile.lab_defaults()
    derived = resolve_context_budget_policy(env_default)
    assert derived.max_chars >= 4000


def test_resolve_context_manager_from_environment() -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    assembly = TaskContextAssemblyOptions(
        max_prior_chars=900,
        summary_tier=ContextSummaryTier.STRUCTURED_ONLY,
    )
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "context_profile": ContextProfile(
                assembly_options=assembly,
                budget_policy=ContextBudgetPolicy(max_chars=5_000, max_tokens_estimate=1_200),
            ),
        }
    )
    manager = resolve_context_manager_from_environment(env)

    assert isinstance(manager, ContextManager)
    assert manager._budget_policy.max_tokens_estimate == 1_200  # noqa: SLF001
    assert manager._default_policy.summary_tier == ContextSummaryTier.STRUCTURED_ONLY  # noqa: SLF001


def test_merge_task_context_options_from_environment() -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    assembly = TaskContextAssemblyOptions(summary_tier=ContextSummaryTier.MINIMAL)
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={"context_profile": ContextProfile(assembly_options=assembly)},
    )
    options = TaskExecutionOptions()
    merged = merge_task_context_options_from_environment(options, env)

    assert merged.context.summary_tier == ContextSummaryTier.MINIMAL
    assert merged.governance == options.governance


def test_default_task_execution_options_for_environment() -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    assembly = TaskContextAssemblyOptions(max_prior_chars=512)
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={"context_profile": ContextProfile(assembly_options=assembly)},
    )
    options = default_task_execution_options_for_environment(env)
    assert options.context.max_prior_chars == 512


def test_build_harness_host_runtime_wires_context_manager_from_environment() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings).model_copy(
        update={
            "context_profile": ContextProfile(
                budget_policy=ContextBudgetPolicy(max_tokens_estimate=2_100),
                assembly_options=TaskContextAssemblyOptions(max_prior_chars=700),
            ),
        }
    )
    manifest = build_lab_manifest(settings)
    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        use_in_memory_trace=True,
    )

    manager = runtime.nexus_loop._context_manager  # noqa: SLF001
    assert manager._budget_policy.max_tokens_estimate == 2_100  # noqa: SLF001
    assert manager._default_policy.max_prior_chars == 700  # noqa: SLF001
