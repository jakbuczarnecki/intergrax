# © Artur Czarnecki. All rights reserved.

"""Adaptive Harness Intelligence runtime package (Phase W-ADAPT-0.3, W-ADAPT-1)."""

from __future__ import annotations

from intergrax.runtime.adaptive.contracts import (
    ADAPTIVE_PACKAGE_SCHEMA_VERSION,
    AdaptiveLifecycleMode,
    HarnessOutcomeSignal,
    OutcomeEvalMode,
    ProcessPatternAction,
    ProcessPatternProposal,
    ProfileArtifactType,
    ProfileVersionDraft,
    ProfileVersionRecord,
    ProfileVersionStatus,
    UtilityWeights,
)
from intergrax.runtime.adaptive.cost_normalization import normalize_cost_against_budget
from intergrax.runtime.adaptive.signal_collector import (
    SignalAssemblyInput,
    SignalCollector,
    regression_flags_from_signals,
)
from intergrax.runtime.adaptive.signal_emission import (
    record_runtime_engine_outcome_signal,
    record_task_outcome_signal,
)
from intergrax.runtime.adaptive.signal_store import (
    InMemorySignalStore,
    SignalStore,
    SQLiteSignalStore,
    default_signal_store,
    default_signal_store_path,
)
from intergrax.runtime.adaptive.utility import compute_utility
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveGovernanceReport,
    AdaptiveLoopEnvelope,
    AdaptiveLoopGateResult,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
    build_default_adaptive_proposals,
    evaluate_adaptive_governance,
    evaluate_bounded_adaptive_loop,
)

__all__ = [
    "ADAPTIVE_PACKAGE_SCHEMA_VERSION",
    "AdaptiveAuthorityLevel",
    "AdaptiveGovernanceReport",
    "AdaptiveLifecycleMode",
    "AdaptiveLoopEnvelope",
    "AdaptiveLoopGateResult",
    "AdaptiveLoopKind",
    "AdaptiveLoopProposal",
    "HarnessOutcomeSignal",
    "InMemorySignalStore",
    "OutcomeEvalMode",
    "ProcessPatternAction",
    "ProcessPatternProposal",
    "ProfileArtifactType",
    "ProfileVersionDraft",
    "ProfileVersionRecord",
    "ProfileVersionStatus",
    "SignalAssemblyInput",
    "SignalCollector",
    "SignalStore",
    "SQLiteSignalStore",
    "UtilityWeights",
    "build_default_adaptive_proposals",
    "compute_utility",
    "default_signal_store",
    "default_signal_store_path",
    "evaluate_adaptive_governance",
    "evaluate_bounded_adaptive_loop",
    "normalize_cost_against_budget",
    "record_runtime_engine_outcome_signal",
    "record_task_outcome_signal",
    "regression_flags_from_signals",
]
