# © Artur Czarnecki. All rights reserved.

"""Built-in capability extractors (extend via new classes, not core conditionals)."""

from __future__ import annotations

from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    AreaAnalysisStatus,
    BehavioralComparisonResult,
    OutcomeSourceRef,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
    BehavioralAnalysisSourceRef,
    CapabilityDimensionId,
    CapabilityObservation,
    LimitationObservation,
    ObservationLevel,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.protocol import (
    CapabilityExtractor,
    CapabilityExtractorResult,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)


def _outcome_ref(outcome: ModelQualificationOutcome) -> OutcomeSourceRef:
    return OutcomeSourceRef(
        profile_key=outcome.profile_key,
        qualification_task_id=outcome.qualification_task_id,
        matrix_version=outcome.matrix_version,
        evaluated_at=outcome.evaluated_at,
    )


def _behavioral_ref(
    comparison: BehavioralComparisonResult,
) -> BehavioralAnalysisSourceRef:
    return BehavioralAnalysisSourceRef(
        analysis_task_id=comparison.analysis_task_id,
        scenario_id=comparison.scenario_id,
        matrix_version=comparison.matrix_version,
        analyzed_at=comparison.analyzed_at,
    )


def _exit_level(exit_code: QualificationCliExit) -> tuple[ObservationLevel, str]:
    if exit_code is QualificationCliExit.SUCCESS:
        return (
            ObservationLevel.STRONG,
            f"observed qualification exit code: {exit_code.name}",
        )
    if exit_code is QualificationCliExit.CRITICAL_SAFETY_FAILURE:
        return (
            ObservationLevel.WEAK,
            f"observed qualification exit code: {exit_code.name}",
        )
    return (
        ObservationLevel.MODERATE,
        f"observed qualification exit code: {exit_code.name}",
    )


class QualificationExitCapabilityExtractor:
    extractor_id = "qualification_exit_capability"
    dimension_id = CapabilityDimensionId.QUALIFICATION_EXIT

    def extract(
        self,
        *,
        profile_key: str,
        outcome: ModelQualificationOutcome | None,
        behavioral_comparison: BehavioralComparisonResult | None,
    ) -> CapabilityExtractorResult:
        _ = profile_key, behavioral_comparison
        if outcome is None:
            return CapabilityExtractorResult(capabilities=(), limitations=())
        level, descriptor = _exit_level(outcome.exit_code)
        evidence = (_outcome_ref(outcome),)
        if level is ObservationLevel.STRONG:
            return CapabilityExtractorResult(
                capabilities=(
                    CapabilityObservation(
                        dimension_id=self.dimension_id,
                        extractor_id=self.extractor_id,
                        level=level,
                        factual_descriptor=descriptor,
                        qualification_evidence=evidence,
                        behavioral_evidence=(),
                    ),
                ),
                limitations=(),
            )
        return CapabilityExtractorResult(
            capabilities=(),
            limitations=(
                LimitationObservation(
                    dimension_id=self.dimension_id,
                    extractor_id=self.extractor_id,
                    level=level,
                    factual_descriptor=descriptor,
                    qualification_evidence=evidence,
                    behavioral_evidence=(),
                ),
            ),
        )


class CohortExecutionCapabilityExtractor:
    extractor_id = "cohort_execution_capability"
    dimension_id = CapabilityDimensionId.COHORT_EXECUTION

    def extract(
        self,
        *,
        profile_key: str,
        outcome: ModelQualificationOutcome | None,
        behavioral_comparison: BehavioralComparisonResult | None,
    ) -> CapabilityExtractorResult:
        _ = profile_key, behavioral_comparison
        if outcome is None:
            return CapabilityExtractorResult(capabilities=(), limitations=())
        evidence = (_outcome_ref(outcome),)
        descriptor = f"observed cohort execution status: {outcome.status.value}"
        if outcome.status is CohortExecutionStatus.EXECUTED:
            return CapabilityExtractorResult(
                capabilities=(
                    CapabilityObservation(
                        dimension_id=self.dimension_id,
                        extractor_id=self.extractor_id,
                        level=ObservationLevel.STRONG,
                        factual_descriptor=descriptor,
                        qualification_evidence=evidence,
                        behavioral_evidence=(),
                    ),
                ),
                limitations=(),
            )
        return CapabilityExtractorResult(
            capabilities=(),
            limitations=(
                LimitationObservation(
                    dimension_id=self.dimension_id,
                    extractor_id=self.extractor_id,
                    level=ObservationLevel.WEAK,
                    factual_descriptor=descriptor,
                    qualification_evidence=evidence,
                    behavioral_evidence=(),
                ),
            ),
        )


class SessionLifecycleCapabilityExtractor:
    extractor_id = "session_lifecycle_capability"
    dimension_id = CapabilityDimensionId.SESSION_LIFECYCLE

    def extract(
        self,
        *,
        profile_key: str,
        outcome: ModelQualificationOutcome | None,
        behavioral_comparison: BehavioralComparisonResult | None,
    ) -> CapabilityExtractorResult:
        _ = profile_key, behavioral_comparison
        if outcome is None:
            return CapabilityExtractorResult(capabilities=(), limitations=())
        evidence = (_outcome_ref(outcome),)
        state = outcome.session_state
        if state is None:
            descriptor = "observed session lifecycle: no session state recorded"
            return CapabilityExtractorResult(
                capabilities=(),
                limitations=(
                    LimitationObservation(
                        dimension_id=self.dimension_id,
                        extractor_id=self.extractor_id,
                        level=ObservationLevel.UNKNOWN,
                        factual_descriptor=descriptor,
                        qualification_evidence=evidence,
                        behavioral_evidence=(),
                    ),
                ),
            )
        descriptor = f"observed session lifecycle state: {state.value}"
        if state is QualificationSessionState.FINALIZED:
            level = ObservationLevel.STRONG
            return CapabilityExtractorResult(
                capabilities=(
                    CapabilityObservation(
                        dimension_id=self.dimension_id,
                        extractor_id=self.extractor_id,
                        level=level,
                        factual_descriptor=descriptor,
                        qualification_evidence=evidence,
                        behavioral_evidence=(),
                    ),
                ),
                limitations=(),
            )
        return CapabilityExtractorResult(
            capabilities=(),
            limitations=(
                LimitationObservation(
                    dimension_id=self.dimension_id,
                    extractor_id=self.extractor_id,
                    level=ObservationLevel.MODERATE,
                    factual_descriptor=descriptor,
                    qualification_evidence=evidence,
                    behavioral_evidence=(),
                ),
            ),
        )


class BehavioralPeerAlignmentExtractor:
    extractor_id = "behavioral_peer_alignment"
    dimension_id = CapabilityDimensionId.BEHAVIORAL_PEER_ALIGNMENT

    def extract(
        self,
        *,
        profile_key: str,
        outcome: ModelQualificationOutcome | None,
        behavioral_comparison: BehavioralComparisonResult | None,
    ) -> CapabilityExtractorResult:
        _ = outcome
        if behavioral_comparison is None:
            return CapabilityExtractorResult(capabilities=(), limitations=())
        behavioral_evidence = (_behavioral_ref(behavioral_comparison),)
        capabilities: list[CapabilityObservation] = []
        limitations: list[LimitationObservation] = []
        for finding in behavioral_comparison.findings:
            differs_here = any(
                diff.profile_key_left == profile_key
                or diff.profile_key_right == profile_key
                for diff in finding.differences
            )
            area = finding.comparison_area.value
            if finding.status is AreaAnalysisStatus.UNIFORM:
                capabilities.append(
                    CapabilityObservation(
                        dimension_id=self.dimension_id,
                        extractor_id=self.extractor_id,
                        level=ObservationLevel.NEUTRAL,
                        factual_descriptor=(
                            f"behavioral comparison area {area}: uniform across cohort"
                        ),
                        qualification_evidence=(),
                        behavioral_evidence=behavioral_evidence,
                    )
                )
            elif finding.status is AreaAnalysisStatus.DIFFERS and differs_here:
                limitations.append(
                    LimitationObservation(
                        dimension_id=self.dimension_id,
                        extractor_id=self.extractor_id,
                        level=ObservationLevel.WEAK,
                        factual_descriptor=(
                            f"behavioral comparison area {area}: differs from peer model(s)"
                        ),
                        qualification_evidence=(),
                        behavioral_evidence=behavioral_evidence,
                    )
                )
            elif finding.status is AreaAnalysisStatus.INSUFFICIENT_DATA:
                limitations.append(
                    LimitationObservation(
                        dimension_id=self.dimension_id,
                        extractor_id=self.extractor_id,
                        level=ObservationLevel.UNKNOWN,
                        factual_descriptor=(
                            f"behavioral comparison area {area}: insufficient data"
                        ),
                        qualification_evidence=(),
                        behavioral_evidence=behavioral_evidence,
                    )
                )
        return CapabilityExtractorResult(
            capabilities=tuple(capabilities),
            limitations=tuple(limitations),
        )


def default_capability_extractors() -> tuple[CapabilityExtractor, ...]:
    return (
        QualificationExitCapabilityExtractor(),
        CohortExecutionCapabilityExtractor(),
        SessionLifecycleCapabilityExtractor(),
        BehavioralPeerAlignmentExtractor(),
    )


__all__ = [
    "BehavioralPeerAlignmentExtractor",
    "CohortExecutionCapabilityExtractor",
    "QualificationExitCapabilityExtractor",
    "SessionLifecycleCapabilityExtractor",
    "default_capability_extractors",
]
