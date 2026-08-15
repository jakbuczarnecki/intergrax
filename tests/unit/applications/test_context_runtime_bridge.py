# © Artur Czarnecki. All rights reserved.

"""CTX-1: ContextProfile → RuntimeConfig bridge."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from intergrax.applications._shared.context_runtime_bridge import (
    CONTEXT_ENGINE_PROFILE_METADATA_KEY,
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
from intergrax.runtime.context_lifecycle.contracts import (
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    EphemeralArtifactPersistencePolicy,
    OptimizationArtifactType,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.wiring.context_runtime_bridge import (
    CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY,
    CONTEXT_OPTIMIZATION_POLICY_SOURCE_CONFLICT_REASON,
    LEGACY_COMPRESSION_STRATEGY_UNSUPPORTED_REASON,
    LEGACY_EXPLICIT_POLICY_CONFLICT_REASON,
    LEGACY_HYBRID_UNSUPPORTED_REASON,
    LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY,
    LEGACY_SEMANTIC_COMPRESSION_METADATA_REASON,
    LEGACY_TRUNCATE_OLDEST_UNSUPPORTED_REASON,
    MESSAGE_SEQUENCE_SUMMARIZATION_STRATEGY_ID,
    LegacyCompressionConfigurationError,
    apply_context_optimization_policy_to_runtime_config,
    resolve_context_optimization_policy,
    resolve_context_optimization_policy_from_profile,
)
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _canonical_summarize_policy() -> ContextOptimizationPolicy:
    return ContextOptimizationPolicy(
        policy_version="policy.v1",
        validation_contract_version="validation.v1",
        enabled=True,
        mode=ContextOptimizationMode.EPHEMERAL_ASSEMBLY,
        allow_lossy=True,
        allow_llm_summarization=True,
        allow_artifact_reuse=True,
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        allowed_strategy_ids=(MESSAGE_SEQUENCE_SUMMARIZATION_STRATEGY_ID,),
        ephemeral_artifact_persistence=EphemeralArtifactPersistencePolicy.DO_NOT_PERSIST,
    )


def _config_snapshot(config: RuntimeConfig) -> dict[str, object]:
    return {
        "enable_rag": config.enable_rag,
        "enable_websearch": config.enable_websearch,
        "context_budget_policy": config.context_budget_policy,
        "context_decision_profile": copy.deepcopy(config.context_decision_profile),
        "max_longterm_entries_per_query": config.max_longterm_entries_per_query,
        "metadata": copy.deepcopy(config.metadata),
        "run_budget": config.run_budget,
        "task_context_assembly_options": config.task_context_assembly_options,
    }


def _config_with_sentinels() -> RuntimeConfig:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    config.enable_rag = False
    config.enable_websearch = True
    config.context_budget_policy = ContextBudgetPolicy(max_tokens_estimate=1_111)
    config.context_decision_profile = {"sentinel": "decision"}
    config.max_longterm_entries_per_query = 42
    config.metadata["sentinel"] = "metadata"
    config.run_budget = RunBudget(max_total_tokens=1_111)
    return config


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


def test_apply_context_profile_maps_engine_preset_fields() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    context = ContextProfile(
        engine_preset="codebase",
        engine_ref="lab.context.CodebaseContextEngine",
        context_plugin_ids=["Acme.Codebase", "  "],
    )

    apply_context_profile_to_runtime_config(config, context)

    payload = config.metadata[CONTEXT_ENGINE_PROFILE_METADATA_KEY]
    assert payload["engine_preset"] == "codebase"
    assert payload["engine_ref"] == "lab.context.CodebaseContextEngine"
    assert payload["context_plugin_ids"] == ["acme.codebase"]


def test_apply_context_profiles_from_environment() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ctx.bridge")
    env.context_profile.budget_policy = ContextBudgetPolicy(max_tokens_estimate=1_800)
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)

    apply_context_profiles_from_environment(config, env)

    assert config.context_budget_policy is not None
    assert config.context_budget_policy.max_tokens_estimate == 1_800


def test_legacy_summarize_oldest_maps_to_executable_policy() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    context = ContextProfile(
        semantic_compression_enabled=True,
        default_history_compression="summarize_oldest",
    )

    apply_context_profile_to_runtime_config(config, context)

    policy = config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY]
    expected = _canonical_summarize_policy()
    assert policy == expected
    assert policy.enabled is True
    assert policy.mode is ContextOptimizationMode.EPHEMERAL_ASSEMBLY
    assert policy.allow_lossy is True
    assert policy.allow_llm_summarization is True
    assert policy.allow_artifact_reuse is True
    assert policy.allowed_artifact_types == (OptimizationArtifactType.MESSAGE_SEQUENCE,)
    assert policy.allowed_strategy_ids == (MESSAGE_SEQUENCE_SUMMARIZATION_STRATEGY_ID,)
    assert policy.ephemeral_artifact_persistence is EphemeralArtifactPersistencePolicy.DO_NOT_PERSIST
    assert LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY not in config.metadata


@pytest.mark.parametrize(
    ("strategy", "reason"),
    [
        ("truncate_oldest", LEGACY_TRUNCATE_OLDEST_UNSUPPORTED_REASON),
        ("hybrid", LEGACY_HYBRID_UNSUPPORTED_REASON),
    ],
)
def test_unsupported_legacy_strategy_rejects_before_runtime_config_mutation(
    strategy: str,
    reason: str,
) -> None:
    config = _config_with_sentinels()
    before = _config_snapshot(config)
    context = ContextProfile(
        enable_rag=True,
        enable_websearch=False,
        budget_policy=ContextBudgetPolicy(max_tokens_estimate=9_999),
        decision=ContextDecisionProfile(max_memory_entries_in_context=12),
        assembly_options=TaskContextAssemblyOptions(summary_tier=ContextSummaryTier.SUMMARY_ONLY),
        semantic_compression_enabled=True,
        default_history_compression=strategy,  # type: ignore[arg-type]
    )

    with pytest.raises(LegacyCompressionConfigurationError) as exc_info:
        apply_context_profile_to_runtime_config(config, context)

    assert exc_info.value.reason == reason
    assert _config_snapshot(config) == before


def test_explicit_optimization_policy_conflicts_with_legacy_compression_without_mutation() -> None:
    config = _config_with_sentinels()
    before = _config_snapshot(config)
    explicit = _canonical_summarize_policy()
    context = ContextProfile(
        optimization_policy=explicit,
        semantic_compression_enabled=True,
        default_history_compression="summarize_oldest",
    )

    with pytest.raises(LegacyCompressionConfigurationError) as exc_info:
        apply_context_profile_to_runtime_config(config, context)

    assert exc_info.value.reason == LEGACY_EXPLICIT_POLICY_CONFLICT_REASON
    assert _config_snapshot(config) == before


def test_unknown_legacy_strategy_is_rejected() -> None:
    context = SimpleNamespace(
        optimization_policy=None,
        semantic_compression_enabled=True,
        default_history_compression="unknown",
    )

    with pytest.raises(LegacyCompressionConfigurationError) as exc_info:
        resolve_context_optimization_policy_from_profile(context)  # type: ignore[arg-type]

    assert exc_info.value.reason == LEGACY_COMPRESSION_STRATEGY_UNSUPPORTED_REASON


def test_writer_rejects_malformed_policy_without_metadata_mutation() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    existing = _canonical_summarize_policy()
    config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY] = existing
    config.metadata[LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY] = {"enabled": True}
    before = dict(config.metadata)

    with pytest.raises(ValueError, match="context_optimization_policy must be"):
        apply_context_optimization_policy_to_runtime_config(config, object())  # type: ignore[arg-type]

    assert config.metadata == before


@pytest.mark.parametrize(
    ("direct_policy", "metadata_policy", "expected", "error"),
    [
        (None, None, None, None),
        ("direct", None, "direct", None),
        (None, "metadata", "metadata", None),
        ("equal_a", "equal_b", "direct", None),
        ("policy_a", "policy_b", None, CONTEXT_OPTIMIZATION_POLICY_SOURCE_CONFLICT_REASON),
        ("invalid", "metadata", None, ValueError),
        ("direct", "malformed_metadata", None, ValueError),
        ("direct", "legacy_metadata", None, LEGACY_SEMANTIC_COMPRESSION_METADATA_REASON),
    ],
)
def test_resolve_context_optimization_policy_matrix(
    direct_policy: str | None,
    metadata_policy: str | None,
    expected: str | None,
    error: type[Exception] | str | None,
) -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    canonical = _canonical_summarize_policy()
    direct_value: ContextOptimizationPolicy | object | None
    if direct_policy is None:
        direct_value = None
    elif direct_policy == "invalid":
        direct_value = {"not": "policy"}
    elif direct_policy in {"equal_a", "policy_a", "direct"}:
        direct_value = canonical
    else:
        direct_value = canonical

    if metadata_policy == "metadata":
        config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY] = canonical
    elif metadata_policy == "equal_b":
        config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY] = ContextOptimizationPolicy(
            policy_version=canonical.policy_version,
            validation_contract_version=canonical.validation_contract_version,
            enabled=canonical.enabled,
            mode=canonical.mode,
            allow_lossy=canonical.allow_lossy,
            allow_llm_summarization=canonical.allow_llm_summarization,
            allow_artifact_reuse=canonical.allow_artifact_reuse,
            allowed_artifact_types=canonical.allowed_artifact_types,
            allowed_strategy_ids=canonical.allowed_strategy_ids,
            ephemeral_artifact_persistence=canonical.ephemeral_artifact_persistence,
        )
    elif metadata_policy == "policy_b":
        config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY] = ContextOptimizationPolicy(
            policy_version="policy.v1",
            validation_contract_version="validation.v1",
            enabled=False,
            mode=ContextOptimizationMode.EPHEMERAL_ASSEMBLY,
            allow_lossy=False,
            allow_llm_summarization=False,
            allowed_artifact_types=(),
            allowed_strategy_ids=(),
        )
    elif metadata_policy == "malformed_metadata":
        config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY] = {"enabled": True}
    elif metadata_policy == "legacy_metadata":
        config.metadata[LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY] = {"enabled": True}

    if error is ValueError:
        with pytest.raises(ValueError):
            resolve_context_optimization_policy(config, direct_policy=direct_value)
        return
    if isinstance(error, str):
        with pytest.raises(LegacyCompressionConfigurationError) as exc_info:
            resolve_context_optimization_policy(config, direct_policy=direct_value)
        assert exc_info.value.reason == error
        return

    resolved = resolve_context_optimization_policy(config, direct_policy=direct_value)
    if expected is None:
        assert resolved is None
    elif expected == "direct":
        assert resolved is direct_value
    elif expected == "metadata":
        assert resolved == canonical


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


def test_disabled_legacy_compression_ignores_default_history_compression() -> None:
    context = ContextProfile(
        semantic_compression_enabled=False,
        default_history_compression="truncate_oldest",
    )

    assert resolve_context_optimization_policy_from_profile(context) is None


def _bridge_context_namespace(**overrides: object) -> SimpleNamespace:
    base = {
        "enable_rag": True,
        "enable_websearch": False,
        "budget_policy": ContextBudgetPolicy(max_tokens_estimate=9_999),
        "assembly_options": TaskContextAssemblyOptions(
            summary_tier=ContextSummaryTier.SUMMARY_ONLY
        ),
        "decision": ContextDecisionProfile(max_memory_entries_in_context=12),
        "drift_monitoring_enabled": False,
        "drift_alert_threshold": 0.35,
        "semantic_compression_enabled": False,
        "default_history_compression": "summarize_oldest",
        "engine_preset": "default",
        "engine_ref": None,
        "context_plugin_ids": [],
        "optimization_policy": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "malformed_policy",
    [object(), {"not": "policy"}, "policy"],
)
def test_malformed_explicit_policy_rejects_before_runtime_config_mutation(
    malformed_policy: object,
) -> None:
    config = _config_with_sentinels()
    before = _config_snapshot(config)
    context = _bridge_context_namespace(optimization_policy=malformed_policy)

    with pytest.raises(ValueError, match="optimization_policy must be"):
        apply_context_profile_to_runtime_config(config, context)  # type: ignore[arg-type]

    assert _config_snapshot(config) == before


def test_malformed_explicit_policy_with_legacy_enabled_raises_value_error() -> None:
    config = _config_with_sentinels()
    before = _config_snapshot(config)
    context = _bridge_context_namespace(
        optimization_policy=object(),
        semantic_compression_enabled=True,
        default_history_compression="summarize_oldest",
    )

    with pytest.raises(ValueError, match="optimization_policy must be"):
        apply_context_profile_to_runtime_config(config, context)  # type: ignore[arg-type]

    assert _config_snapshot(config) == before


def test_valid_explicit_policy_preserves_identity_through_full_bridge() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    explicit = _canonical_summarize_policy()
    decision = ContextDecisionProfile(
        include_session_history=False,
        prefer_longterm_memory=True,
        prefer_rag_when_enabled=False,
        max_memory_entries_in_context=12,
    )
    budget = ContextBudgetPolicy(max_chars=8_000, max_tokens_estimate=2_000)
    assembly = TaskContextAssemblyOptions(summary_tier=ContextSummaryTier.SUMMARY_ONLY)
    context = ContextProfile(
        enable_rag=False,
        enable_websearch=True,
        budget_policy=budget,
        decision=decision,
        assembly_options=assembly,
        optimization_policy=explicit,
        semantic_compression_enabled=False,
    )

    apply_context_profile_to_runtime_config(config, context)

    assert config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY] is explicit
    assert LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY not in config.metadata
    assert config.enable_rag is False
    assert config.enable_websearch is True
    assert config.context_budget_policy == budget
    assert config.task_context_assembly_options == assembly
    assert config.context_decision_profile == decision.model_dump(mode="json")
    assert config.max_longterm_entries_per_query == 12
