# © Artur Czarnecki. All rights reserved.

"""Model capability profile orchestration (DS-E2E-15J-L3)."""

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    ModelIdentityRef,
    OutcomeSourceRef,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
    BASELINE_TASK_ID,
    BASELINE_VERSION,
    BehavioralAnalysisSourceRef,
    CapabilityProfileBuildRequest,
    CapabilityProfileBuildResult,
    CapabilityProfileBuildStatus,
    ModelCapabilityProfile,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.extractors import (
    default_capability_extractors,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.protocol import (
    CapabilityExtractor,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)


def _outcome_source_ref(outcome: ModelQualificationOutcome) -> OutcomeSourceRef:
    return OutcomeSourceRef(
        profile_key=outcome.profile_key,
        qualification_task_id=outcome.qualification_task_id,
        matrix_version=outcome.matrix_version,
        evaluated_at=outcome.evaluated_at,
    )


def _behavioral_source_ref(
    request: CapabilityProfileBuildRequest,
) -> BehavioralAnalysisSourceRef | None:
    comparison = request.behavioral_comparison
    if comparison is None:
        return None
    return BehavioralAnalysisSourceRef(
        analysis_task_id=comparison.analysis_task_id,
        scenario_id=comparison.scenario_id,
        matrix_version=comparison.matrix_version,
        analyzed_at=comparison.analyzed_at,
    )


def _validate_request(
    request: CapabilityProfileBuildRequest,
) -> CapabilityProfileBuildStatus | None:
    if not request.outcomes:
        return CapabilityProfileBuildStatus.INSUFFICIENT_DATA
    for outcome in request.outcomes:
        if outcome.matrix_version != request.matrix_version:
            return CapabilityProfileBuildStatus.VERSION_MISMATCH
    comparison = request.behavioral_comparison
    if comparison is not None and comparison.matrix_version != request.matrix_version:
        return CapabilityProfileBuildStatus.VERSION_MISMATCH
    return None


def _build_profile_for_model(
    *,
    outcome: ModelQualificationOutcome,
    request: CapabilityProfileBuildRequest,
    extractors: tuple[CapabilityExtractor, ...],
    generated_at: datetime,
    behavioral_source_ref: BehavioralAnalysisSourceRef | None,
) -> ModelCapabilityProfile:
    capabilities = []
    limitations = []
    for extractor in extractors:
        chunk = extractor.extract(
            profile_key=outcome.profile_key,
            outcome=outcome,
            behavioral_comparison=request.behavioral_comparison,
        )
        capabilities.extend(chunk.capabilities)
        limitations.extend(chunk.limitations)

    qualification_versions = (outcome.matrix_version,)
    return ModelCapabilityProfile(
        model_identity=ModelIdentityRef(
            profile_key=outcome.profile_key,
            provider=outcome.provider,
            model_name=outcome.model_name,
        ),
        model_version=outcome.model_name,
        matrix_version=request.matrix_version,
        baseline_task_id=BASELINE_TASK_ID,
        baseline_version=BASELINE_VERSION,
        generated_at=generated_at,
        qualification_source_refs=(_outcome_source_ref(outcome),),
        behavioral_source_ref=behavioral_source_ref,
        source_qualification_versions=qualification_versions,
        source_behavioral_matrix_version=(
            behavioral_source_ref.matrix_version if behavioral_source_ref else None
        ),
        capabilities=tuple(capabilities),
        limitations=tuple(limitations),
    )


def build_model_capability_profiles(
    request: CapabilityProfileBuildRequest,
    extractors: tuple[CapabilityExtractor, ...] | None = None,
    *,
    generated_at: datetime | None = None,
) -> CapabilityProfileBuildResult:
    stamp = generated_at or datetime.now(tz=UTC)
    validation_status = _validate_request(request)
    active_extractors = (
        extractors if extractors is not None else default_capability_extractors()
    )
    behavioral_ref = _behavioral_source_ref(request)

    if validation_status is not None:
        return CapabilityProfileBuildResult(
            build_task_id=BASELINE_TASK_ID,
            baseline_version=BASELINE_VERSION,
            matrix_version=request.matrix_version,
            generated_at=stamp,
            status=validation_status,
            profiles=(),
        )

    profiles = tuple(
        _build_profile_for_model(
            outcome=outcome,
            request=request,
            extractors=active_extractors,
            generated_at=stamp,
            behavioral_source_ref=behavioral_ref,
        )
        for outcome in request.outcomes
    )
    return CapabilityProfileBuildResult(
        build_task_id=BASELINE_TASK_ID,
        baseline_version=BASELINE_VERSION,
        matrix_version=request.matrix_version,
        generated_at=stamp,
        status=CapabilityProfileBuildStatus.COMPLETE,
        profiles=profiles,
    )


__all__ = ["build_model_capability_profiles"]
