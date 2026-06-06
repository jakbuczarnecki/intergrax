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
from intergrax.runtime.adaptive.adaptation_engine import AdaptationEngine
from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor, ShadowAllocationResult
from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationEngineContext,
    AdaptationEngineRunResult,
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
    BanditArmState,
)
from intergrax.runtime.adaptive.adaptation_scheduler import AdaptationScheduler
from intergrax.runtime.adaptive.adaptation_sub_engine import AdaptationSubEngine
from intergrax.runtime.adaptive.bandit_state_store import (
    BanditStateStore,
    InMemoryBanditStateStore,
    SQLiteBanditStateStore,
    default_bandit_store_path,
)
from intergrax.runtime.adaptive.cost_anomaly_bridge import proposals_from_cost_anomalies
from intergrax.runtime.adaptive.evaluation_feedback_engine import EvaluationFeedbackEngine
from intergrax.runtime.adaptive.execution_strategy_engine import ExecutionStrategyEngine
from intergrax.runtime.adaptive.governance_pipeline import (
    AdaptationGovernancePipeline,
    validate_evaluation_assets_bundle,
)
from intergrax.runtime.adaptive.policy_learning_engine import PolicyLearningEngine
from intergrax.runtime.adaptive.proposal_builder import ProposalBuilder
from intergrax.runtime.adaptive.proposal_cooldown_store import (
    InMemoryProposalCooldownStore,
    ProposalCooldownStore,
)
from intergrax.runtime.adaptive.proposal_store import (
    InMemoryProposalStore,
    ProposalStore,
    SQLiteProposalStore,
    default_proposal_store_path,
)
from intergrax.runtime.adaptive.profile_lifecycle import (
    ProfileLifecycleTransitionError,
    ProfileVersionLifecycleManager,
    validate_profile_transition,
)
from intergrax.runtime.adaptive.profile_promotion import (
    ProfilePromotionDecision,
    ProfilePromotionEvidenceBundle,
    evaluate_profile_promotion,
)
from intergrax.runtime.adaptive.profile_rag_router import (
    ProfileAwareQueryRouter,
    apply_rag_profile_version,
)
from intergrax.runtime.adaptive.profile_version_store import (
    InMemoryProfileVersionStore,
    ProfileVersionStore,
    SQLiteProfileVersionStore,
    default_profile_version_store_path,
)
from intergrax.runtime.adaptive.routing_tuning_engine import RoutingTuningEngine
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
    "AdaptationEngine",
    "AdaptationExecutor",
    "AdaptationEngineContext",
    "AdaptationEngineRunResult",
    "AdaptationGovernancePipeline",
    "AdaptationProposalCandidate",
    "AdaptationProposalPackage",
    "AdaptationScheduler",
    "AdaptationSubEngine",
    "AdaptiveAuthorityLevel",
    "AdaptiveGovernanceReport",
    "AdaptiveLifecycleMode",
    "AdaptiveLoopEnvelope",
    "AdaptiveLoopGateResult",
    "AdaptiveLoopKind",
    "AdaptiveLoopProposal",
    "BanditArmState",
    "BanditStateStore",
    "EvaluationFeedbackEngine",
    "ExecutionStrategyEngine",
    "HarnessOutcomeSignal",
    "InMemoryBanditStateStore",
    "InMemoryProfileVersionStore",
    "InMemoryProposalCooldownStore",
    "InMemoryProposalStore",
    "InMemorySignalStore",
    "OutcomeEvalMode",
    "PolicyLearningEngine",
    "ProcessPatternAction",
    "ProcessPatternProposal",
    "ProfileArtifactType",
    "ProfileAwareQueryRouter",
    "ProfileLifecycleTransitionError",
    "ProfilePromotionDecision",
    "ProfilePromotionEvidenceBundle",
    "ProfileVersionLifecycleManager",
    "ProfileVersionStore",
    "ProfileVersionDraft",
    "ProfileVersionRecord",
    "ProfileVersionStatus",
    "ProposalBuilder",
    "ProposalCooldownStore",
    "ProposalStore",
    "RoutingTuningEngine",
    "SignalAssemblyInput",
    "SignalCollector",
    "SignalStore",
    "SQLiteBanditStateStore",
    "ShadowAllocationResult",
    "SQLiteProfileVersionStore",
    "SQLiteSignalStore",
    "UtilityWeights",
    "SQLiteProposalStore",
    "apply_rag_profile_version",
    "build_default_adaptive_proposals",
    "compute_utility",
    "default_bandit_store_path",
    "default_profile_version_store_path",
    "default_proposal_store_path",
    "default_signal_store",
    "default_signal_store_path",
    "evaluate_adaptive_governance",
    "evaluate_bounded_adaptive_loop",
    "evaluate_profile_promotion",
    "normalize_cost_against_budget",
    "proposals_from_cost_anomalies",
    "record_runtime_engine_outcome_signal",
    "record_task_outcome_signal",
    "regression_flags_from_signals",
    "validate_evaluation_assets_bundle",
    "validate_profile_transition",
]
