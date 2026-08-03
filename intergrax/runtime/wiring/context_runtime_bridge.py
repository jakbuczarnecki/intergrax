# © Artur Czarnecki. All rights reserved.

"""Map host context profile fields to RuntimeConfig."""

from __future__ import annotations

from typing import Literal, Protocol

from intergrax.runtime.context_lifecycle.contracts import (
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    EphemeralArtifactPersistencePolicy,
    OptimizationArtifactType,
)
from intergrax.runtime.nexus.config import RuntimeConfig

CONTEXT_ENGINE_PROFILE_METADATA_KEY = "context_engine_profile.v1"
CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY = "context_optimization_policy.v1"
CONTEXT_OPTIMIZATION_POLICY_HANDLE = "context_optimization_policy"
LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY = "semantic_compression.v1"
MESSAGE_SEQUENCE_SUMMARIZATION_STRATEGY_ID = "message_sequence_summarization.v1"
DEFAULT_OPTIMIZATION_POLICY_VERSION = "policy.v1"
DEFAULT_OPTIMIZATION_VALIDATION_CONTRACT_VERSION = "validation.v1"

LEGACY_EXPLICIT_POLICY_CONFLICT_REASON = (
    "legacy_compression_conflicts_with_explicit_policy"
)
LEGACY_TRUNCATE_OLDEST_UNSUPPORTED_REASON = (
    "legacy_truncate_oldest_not_supported_by_canonical_ucl"
)
LEGACY_HYBRID_UNSUPPORTED_REASON = (
    "legacy_hybrid_not_supported_by_canonical_ucl"
)
LEGACY_COMPRESSION_STRATEGY_UNSUPPORTED_REASON = (
    "legacy_compression_strategy_not_supported_by_canonical_ucl"
)
LEGACY_SEMANTIC_COMPRESSION_METADATA_REASON = (
    "legacy_semantic_compression_metadata_not_authoritative"
)
CONTEXT_OPTIMIZATION_POLICY_SOURCE_CONFLICT_REASON = (
    "context_optimization_policy_source_conflict"
)

LegacyHistoryCompression = Literal["truncate_oldest", "summarize_oldest", "hybrid"]


class LegacyCompressionConfigurationError(ValueError):
    """Raised when legacy compression settings cannot be migrated to UCL policy."""

    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _ContextProfileBridgeSource(Protocol):
    enable_rag: bool
    enable_websearch: bool
    budget_policy: object | None
    assembly_options: object
    decision: object
    drift_monitoring_enabled: bool
    drift_alert_threshold: float
    semantic_compression_enabled: bool
    default_history_compression: LegacyHistoryCompression
    engine_preset: str
    engine_ref: str | None
    context_plugin_ids: list[str]
    optimization_policy: ContextOptimizationPolicy | None


def resolve_context_optimization_policy_from_profile(
    context: _ContextProfileBridgeSource,
) -> ContextOptimizationPolicy | None:
    """Map explicit or legacy profile compression controls to canonical UCL policy."""
    explicit_policy = context.optimization_policy
    if explicit_policy is not None:
        if not isinstance(explicit_policy, ContextOptimizationPolicy):
            raise ValueError(
                "optimization_policy must be "
                "ContextOptimizationPolicy or None"
            )

        if context.semantic_compression_enabled:
            raise LegacyCompressionConfigurationError(
                LEGACY_EXPLICIT_POLICY_CONFLICT_REASON
            )

        return explicit_policy

    if not context.semantic_compression_enabled:
        return None

    return _legacy_history_compression_to_policy(context.default_history_compression)


def _legacy_history_compression_to_policy(
    strategy: str,
) -> ContextOptimizationPolicy:
    if strategy == "summarize_oldest":
        return ContextOptimizationPolicy(
            policy_version=DEFAULT_OPTIMIZATION_POLICY_VERSION,
            validation_contract_version=DEFAULT_OPTIMIZATION_VALIDATION_CONTRACT_VERSION,
            enabled=True,
            mode=ContextOptimizationMode.EPHEMERAL_ASSEMBLY,
            allow_lossy=True,
            allow_llm_summarization=True,
            allow_artifact_reuse=True,
            allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
            allowed_strategy_ids=(MESSAGE_SEQUENCE_SUMMARIZATION_STRATEGY_ID,),
            ephemeral_artifact_persistence=(
                EphemeralArtifactPersistencePolicy.DO_NOT_PERSIST
            ),
        )
    if strategy == "truncate_oldest":
        raise LegacyCompressionConfigurationError(
            LEGACY_TRUNCATE_OLDEST_UNSUPPORTED_REASON
        )
    if strategy == "hybrid":
        raise LegacyCompressionConfigurationError(
            LEGACY_HYBRID_UNSUPPORTED_REASON
        )
    raise LegacyCompressionConfigurationError(
        LEGACY_COMPRESSION_STRATEGY_UNSUPPORTED_REASON
    )


def apply_context_optimization_policy_to_runtime_config(
    config: RuntimeConfig,
    policy: ContextOptimizationPolicy | None,
) -> None:
    """Persist canonical optimization policy on runtime config metadata."""
    if policy is not None and not isinstance(policy, ContextOptimizationPolicy):
        raise ValueError(
            "context_optimization_policy must be "
            "ContextOptimizationPolicy or None"
        )
    config.metadata.pop(LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY, None)
    if policy is None:
        config.metadata.pop(CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY, None)
        return
    config.metadata[CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY] = policy


def optimization_policy_from_runtime_config(
    runtime_config: RuntimeConfig,
) -> ContextOptimizationPolicy | None:
    """Read canonical optimization policy previously applied by the bridge."""
    if LEGACY_SEMANTIC_COMPRESSION_METADATA_KEY in runtime_config.metadata:
        raise LegacyCompressionConfigurationError(
            LEGACY_SEMANTIC_COMPRESSION_METADATA_REASON
        )
    raw = runtime_config.metadata.get(CONTEXT_OPTIMIZATION_POLICY_METADATA_KEY)
    if raw is None:
        return None
    if not isinstance(raw, ContextOptimizationPolicy):
        raise ValueError("context_optimization_policy.v1 must be ContextOptimizationPolicy")
    return raw


def resolve_context_optimization_policy(
    runtime_config: RuntimeConfig,
    *,
    direct_policy: object | None,
) -> ContextOptimizationPolicy | None:
    """Resolve canonical optimization policy from runtime metadata and direct handle."""
    metadata_policy = optimization_policy_from_runtime_config(runtime_config)
    if direct_policy is None:
        return metadata_policy
    if not isinstance(direct_policy, ContextOptimizationPolicy):
        raise ValueError(
            "context_optimization_policy handle "
            "must be ContextOptimizationPolicy"
        )
    if metadata_policy is None:
        return direct_policy
    if direct_policy != metadata_policy:
        raise LegacyCompressionConfigurationError(
            CONTEXT_OPTIMIZATION_POLICY_SOURCE_CONFLICT_REASON
        )
    return direct_policy


def apply_context_profile_to_runtime_config(
    config: RuntimeConfig,
    context: _ContextProfileBridgeSource,
) -> RuntimeConfig:
    """Apply ``ContextProfile`` budget, assembly, and decision fields."""
    optimization_policy = resolve_context_optimization_policy_from_profile(context)
    config.enable_rag = context.enable_rag
    config.enable_websearch = context.enable_websearch
    if context.budget_policy is not None:
        config.context_budget_policy = context.budget_policy
    config.task_context_assembly_options = context.assembly_options
    config.context_decision_profile = context.decision.model_dump(mode="json")
    if context.drift_monitoring_enabled:
        config.metadata["context_drift_monitoring.v1"] = {
            "enabled": True,
            "alert_threshold": context.drift_alert_threshold,
        }
    apply_context_optimization_policy_to_runtime_config(config, optimization_policy)
    if context.decision.max_memory_entries_in_context != config.max_longterm_entries_per_query:
        config.max_longterm_entries_per_query = context.decision.max_memory_entries_in_context
    config.metadata[CONTEXT_ENGINE_PROFILE_METADATA_KEY] = {
        "engine_preset": context.engine_preset,
        "engine_ref": context.engine_ref,
        "context_plugin_ids": list(context.context_plugin_ids),
    }
    derive_run_budget_from_context_policy(config)
    return config


def derive_run_budget_from_context_policy(config: RuntimeConfig) -> RuntimeConfig:
    """Mirror context token budget into Nexus ``RunBudget`` when unset."""
    if config.context_budget_policy is not None and config.run_budget is None:
        from intergrax.runtime.nexus.budget.budget_models import RunBudget

        policy = config.context_budget_policy
        config.run_budget = RunBudget(max_total_tokens=policy.max_tokens_estimate)
    return config
