# © Artur Czarnecki. All rights reserved.

"""Built-in optimization pattern analyzers (extend via new classes)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    AnalysisPayloadKind,
    AnalyticsResultStatus,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    ConfidenceLevel,
    DecisionOptimizationContext,
    DetectedOptimizationPattern,
    OptimizationArea,
    OptimizationDataSourceKind,
    OptimizationDataSourceRef,
)


def _analytics_ref(result_index: int) -> OptimizationDataSourceRef:
    return OptimizationDataSourceRef(
        source_kind=OptimizationDataSourceKind.ANALYTICS_RESULT,
        reference_id=f"analytics:{result_index}",
    )


class GovernanceFrictionPatternAnalyzer:
    """Detects elevated governance blocks/require-approval from analytics facts."""

    analyzer_id = "governance_friction"
    analyzer_version = "1"
    _block_ratio_threshold = 0.2

    def detect_patterns(
        self,
        context: DecisionOptimizationContext,
    ) -> tuple[DetectedOptimizationPattern, ...]:
        patterns: list[DetectedOptimizationPattern] = []
        for index, result in enumerate(context.analytics_results):
            if result.payload_kind is not AnalysisPayloadKind.GOVERNANCE:
                continue
            if result.status is not AnalyticsResultStatus.COMPLETE:
                continue
            payload = result.governance_payload
            if payload is None:
                continue
            total = payload.allow + payload.block + payload.require_approval
            if total == 0:
                continue
            friction = payload.block + payload.require_approval
            ratio = friction / total
            if ratio < self._block_ratio_threshold:
                continue
            decision_ids = result.audit.decision_ids
            patterns.append(
                DetectedOptimizationPattern(
                    pattern_id=f"{self.analyzer_id}:{index}",
                    analyzer_id=self.analyzer_id,
                    analyzer_version=self.analyzer_version,
                    optimization_area=OptimizationArea.GOVERNANCE_FRICTION,
                    summary=(
                        f"Governance friction ratio {ratio:.2f} "
                        f"({friction}/{total} block or require approval)."
                    ),
                    source_decision_ids=decision_ids,
                    data_source_refs=(_analytics_ref(index),),
                    confidence=ConfidenceLevel.HIGH
                    if ratio >= 0.5
                    else ConfidenceLevel.MEDIUM,
                )
            )
        return tuple(patterns)


class OutcomeReliabilityPatternAnalyzer:
    """Detects elevated failed or blocked terminal outcomes."""

    analyzer_id = "outcome_reliability"
    analyzer_version = "1"
    _adverse_ratio_threshold = 0.25

    def detect_patterns(
        self,
        context: DecisionOptimizationContext,
    ) -> tuple[DetectedOptimizationPattern, ...]:
        patterns: list[DetectedOptimizationPattern] = []
        for index, result in enumerate(context.analytics_results):
            if result.payload_kind is not AnalysisPayloadKind.OUTCOME:
                continue
            if result.status is not AnalyticsResultStatus.COMPLETE:
                continue
            payload = result.outcome_payload
            if payload is None:
                continue
            total = (
                payload.completed
                + payload.failed
                + payload.blocked
                + payload.in_progress
            )
            if total == 0:
                continue
            adverse = payload.failed + payload.blocked
            ratio = adverse / total
            if ratio < self._adverse_ratio_threshold:
                continue
            patterns.append(
                DetectedOptimizationPattern(
                    pattern_id=f"{self.analyzer_id}:{index}",
                    analyzer_id=self.analyzer_id,
                    analyzer_version=self.analyzer_version,
                    optimization_area=OptimizationArea.OUTCOME_RELIABILITY,
                    summary=(
                        f"Adverse outcome ratio {ratio:.2f} "
                        f"({adverse}/{total} failed or blocked)."
                    ),
                    source_decision_ids=result.audit.decision_ids,
                    data_source_refs=(_analytics_ref(index),),
                    confidence=ConfidenceLevel.HIGH
                    if ratio >= 0.4
                    else ConfidenceLevel.MEDIUM,
                )
            )
        return tuple(patterns)


class ModelCapabilityPatternAnalyzer:
    """Surfaces capability/limitation imbalance across model profiles."""

    analyzer_id = "model_capability"
    analyzer_version = "1"

    def detect_patterns(
        self,
        context: DecisionOptimizationContext,
    ) -> tuple[DetectedOptimizationPattern, ...]:
        patterns: list[DetectedOptimizationPattern] = []
        for index, profile in enumerate(context.capability_profiles):
            limitation_count = len(profile.limitations)
            capability_count = len(profile.capabilities)
            if limitation_count <= capability_count:
                continue
            model_key = profile.model_identity.profile_key
            patterns.append(
                DetectedOptimizationPattern(
                    pattern_id=f"{self.analyzer_id}:{model_key}",
                    analyzer_id=self.analyzer_id,
                    analyzer_version=self.analyzer_version,
                    optimization_area=OptimizationArea.MODEL_CAPABILITY,
                    summary=(
                        f"Model {model_key} has more recorded limitations "
                        f"({limitation_count}) than capabilities ({capability_count})."
                    ),
                    source_decision_ids=(),
                    data_source_refs=(
                        OptimizationDataSourceRef(
                            source_kind=OptimizationDataSourceKind.CAPABILITY_PROFILE,
                            reference_id=model_key,
                        ),
                    ),
                    confidence=ConfidenceLevel.MEDIUM,
                )
            )
        return tuple(patterns)


def default_pattern_analyzers() -> tuple[
    GovernanceFrictionPatternAnalyzer,
    OutcomeReliabilityPatternAnalyzer,
    ModelCapabilityPatternAnalyzer,
]:
    return (
        GovernanceFrictionPatternAnalyzer(),
        OutcomeReliabilityPatternAnalyzer(),
        ModelCapabilityPatternAnalyzer(),
    )


__all__ = [
    "GovernanceFrictionPatternAnalyzer",
    "ModelCapabilityPatternAnalyzer",
    "OutcomeReliabilityPatternAnalyzer",
    "default_pattern_analyzers",
]
