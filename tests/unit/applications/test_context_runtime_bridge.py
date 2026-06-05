# © Artur Czarnecki. All rights reserved.

"""CTX-1: ContextProfile → RuntimeConfig bridge."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.context_runtime_bridge import (
    apply_context_profile_to_runtime_config,
    apply_context_profiles_from_environment,
    derive_run_budget_from_context_policy,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextDecisionProfile,
    ContextProfile,
)
from intergrax.contracts.context_assembly import ContextSummaryTier, TaskContextAssemblyOptions
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_apply_context_profile_maps_budget_and_decision() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    budget = ContextBudgetPolicy(max_chars=8_000, max_tokens_estimate=2_000)
    decision = ContextDecisionProfile(
        include_session_history=False,
        prefer_longterm_memory=True,
        prefer_rag_when_enabled=False,
        max_memory_entries_in_context=12,
    )
    assembly = TaskContextAssemblyOptions(summary_tier=ContextSummaryTier.SUMMARY_ONLY)
    context = ContextProfile(
        enable_rag=False,
        enable_websearch=True,
        budget_policy=budget,
        decision=decision,
        assembly_options=assembly,
    )

    apply_context_profile_to_runtime_config(config, context)

    assert config.enable_rag is False
    assert config.enable_websearch is True
    assert config.context_budget_policy == budget
    assert config.task_context_assembly_options == assembly
    assert config.context_decision_profile == decision.model_dump(mode="json")
    assert config.max_longterm_entries_per_query == 12
    assert config.run_budget is not None
    assert config.run_budget.max_total_tokens == 2_000


def test_derive_run_budget_from_context_policy() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    config.context_budget_policy = ContextBudgetPolicy(max_tokens_estimate=3_500)

    derive_run_budget_from_context_policy(config)

    assert config.run_budget is not None
    assert config.run_budget.max_total_tokens == 3_500


def test_apply_context_profiles_from_environment() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ctx.bridge")
    env.context_profile.budget_policy = ContextBudgetPolicy(max_tokens_estimate=1_800)
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)

    apply_context_profiles_from_environment(config, env)

    assert config.context_budget_policy is not None
    assert config.context_budget_policy.max_tokens_estimate == 1_800
