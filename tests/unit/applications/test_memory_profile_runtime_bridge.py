# © Artur Czarnecki. All rights reserved.

"""MEM-1.5: ApplicationEnvironmentProfile memory/context → RuntimeConfig bridge."""

from __future__ import annotations

import pytest

from intergrax.agents.reference_harness import LabHarnessContext, default_reference_harness
from intergrax.applications._shared.memory_runtime_bridge import (
    apply_environment_profiles_to_runtime_config,
    apply_memory_profile_to_runtime_config,
    apply_context_profile_to_runtime_config,
)
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    ContextDecisionProfile,
    ContextProfile,
    MemoryProfile,
)
from intergrax.contracts.context_assembly import ContextSummaryTier, TaskContextAssemblyOptions
from intergrax.runtime.context_lifecycle.contracts import (
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    OptimizationArtifactType,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.wiring.context_runtime_bridge import (
    CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY,
    LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY,
    LegacyCompressionConfigurationError,
    resolve_context_optimization_policy_from_profile,
)
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _runtime_request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-mem",
        agent_id="agent_mem",
        user_id="user_mem",
        session_id="session_mem",
        message="memory bridge probe",
    )


def test_apply_memory_profile_maps_toggles_and_policy() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    memory = MemoryProfile(
        enable_user_memory=True,
        enable_org_memory=False,
        enable_long_term_memory=True,
        enable_task_memory=True,
        retention_days=30,
        scope_boundary="tenant",
    )

    apply_memory_profile_to_runtime_config(config, memory)

    assert config.enable_user_profile_memory is True
    assert config.enable_org_profile_memory is False
    assert config.enable_user_longterm_memory is True
    assert config.enable_task_memory is True
    assert config.memory_retention_days == 30
    assert config.memory_scope_boundary == "tenant"


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


def test_apply_environment_profiles_derives_run_budget_from_context_budget() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.bridge")
    env.context_profile.budget_policy = ContextBudgetPolicy(max_tokens_estimate=3_500)

    apply_environment_profiles_to_runtime_config(config, env)

    assert config.run_budget is not None
    assert config.run_budget.max_total_tokens == 3_500


def test_materialize_runtime_config_round_trip_from_environment_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.materialize")
    env.memory_profile = MemoryProfile(
        enable_user_memory=True,
        enable_org_memory=True,
        enable_long_term_memory=True,
        enable_task_memory=True,
        retention_days=14,
        scope_boundary="tenant",
    )
    env.context_profile = ContextProfile(
        enable_rag=True,
        enable_websearch=False,
        budget_policy=ContextBudgetPolicy(max_tokens_estimate=1_500),
        decision=ContextDecisionProfile(max_memory_entries_in_context=5),
    )
    harness = LabHarnessContext(
        policy_bundle=RuntimePolicyBundle(),
        strict_harness=True,
    )

    config = materialize_runtime_config(
        _runtime_request(),
        harness,
        env,
        llm_adapter=FakeLLMAdapter(fixed_text="bridge-ok"),
    )

    assert config.enable_user_profile_memory is True
    assert config.enable_org_profile_memory is True
    assert config.enable_user_longterm_memory is True
    assert config.enable_task_memory is True
    assert config.memory_retention_days == 14
    assert config.enable_rag is True
    assert config.enable_websearch is False
    assert config.context_budget_policy is not None
    assert config.context_budget_policy.max_tokens_estimate == 1_500
    assert config.max_longterm_entries_per_query == 5
    assert config.run_budget is not None
    assert config.run_budget.max_total_tokens == 1_500
    assert config.production_mode is True


def test_materialize_runtime_config_uses_default_harness_when_not_strict() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.harness-default")
    harness = default_reference_harness()

    config = materialize_runtime_config(_runtime_request(), harness, env)

    assert config.llm_adapter is not None
    assert config.enable_task_memory is True


def test_legacy_semantic_compression_maps_to_canonical_policy() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    context = ContextProfile(
        semantic_compression_enabled=True,
        default_history_compression="summarize_oldest",
    )

    apply_context_profile_to_runtime_config(config, context)

    policy = config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY]
    assert isinstance(policy, ContextOptimizationPolicy)
    assert policy.enabled is True
    assert policy.allow_llm_summarization is True
    assert policy.allow_lossy is True
    assert LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY not in config.metadata


def test_legacy_truncate_strategy_disables_summarizer() -> None:
    context = ContextProfile(
        semantic_compression_enabled=True,
        default_history_compression="truncate_oldest",
    )

    policy = resolve_context_optimization_policy_from_profile(context)

    assert policy is not None
    assert policy.allow_llm_summarization is False
    assert policy.allowed_strategy_ids == ()


def test_explicit_optimization_policy_conflicts_with_legacy_compression() -> None:
    explicit = ContextOptimizationPolicy(
        policy_version="policy.v1",
        validation_contract_version="validation.v1",
        enabled=True,
        mode=ContextOptimizationMode.EPHEMERAL_ASSEMBLY,
        allow_lossy=True,
        allow_llm_summarization=True,
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        allowed_strategy_ids=("message_sequence_summarization.v1",),
    )
    context = ContextProfile(
        optimization_policy=explicit,
        semantic_compression_enabled=True,
    )

    with pytest.raises(LegacyCompressionConfigurationError):
        resolve_context_optimization_policy_from_profile(context)


def test_canonical_budget_affects_run_budget_not_message_slicing() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    context = ContextProfile(
        budget_policy=ContextBudgetPolicy(max_tokens_estimate=2_400),
    )

    apply_context_profile_to_runtime_config(config, context)

    assert config.context_budget_policy is not None
    assert config.context_budget_policy.max_tokens_estimate == 2_400
    assert config.run_budget is not None
    assert config.run_budget.max_total_tokens == 2_400
