# © Artur Czarnecki. All rights reserved.

"""Tier-3 context profile helpers (CE-11.1, CE-11.4)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ContextProfile
from intergrax.contracts.context_assembly import ContextSummaryTier, TaskContextAssemblyOptions
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy


def production_context_profile() -> ContextProfile:
    return ContextProfile(
        engine_preset="default",
        context_plugin_ids=["intergrax.builtin"],
        assembly_options=TaskContextAssemblyOptions(max_prior_chars=16_000),
        budget_policy=ContextBudgetPolicy(max_chars=16_000, max_tokens_estimate=4_000),
    )


def codebase_context_profile() -> ContextProfile:
    return ContextProfile(
        engine_preset="codebase",
        context_plugin_ids=["intergrax.builtin"],
        assembly_options=TaskContextAssemblyOptions(max_prior_chars=24_000),
        budget_policy=ContextBudgetPolicy(max_chars=24_000, max_tokens_estimate=6_000),
    )


def regulated_minimal_context_profile() -> ContextProfile:
    return ContextProfile(
        engine_preset="regulated_minimal",
        context_plugin_ids=["intergrax.builtin"],
        assembly_options=TaskContextAssemblyOptions(
            max_prior_chars=4_000,
            summary_tier=ContextSummaryTier.MINIMAL,
        ),
        budget_policy=ContextBudgetPolicy(max_chars=4_000, max_tokens_estimate=1_000),
        enable_websearch=False,
    )


def explore_child_context_profile() -> ContextProfile:
    return ContextProfile(
        engine_preset="explore_child",
        context_plugin_ids=["intergrax.builtin"],
        assembly_options=TaskContextAssemblyOptions(max_prior_chars=2_000),
        budget_policy=ContextBudgetPolicy(max_chars=2_000, max_tokens_estimate=512),
    )
