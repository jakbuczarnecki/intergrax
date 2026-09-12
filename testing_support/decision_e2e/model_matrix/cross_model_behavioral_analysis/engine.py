# © Artur Czarnecki. All rights reserved.

"""Cross-model behavioral analysis orchestration (DS-E2E-15J-L2)."""

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.analyzers import (
    default_behavior_analyzers,
)
from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    ANALYSIS_TASK_ID,
    BehavioralComparisonResult,
    CrossModelAnalysisStatus,
    CrossModelBehavioralAnalysisRequest,
    ModelIdentityRef,
    OutcomeSourceRef,
)
from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.protocol import (
    BehaviorAnalyzer,
)


def _model_refs(
    outcomes: tuple,
) -> tuple[ModelIdentityRef, ...]:
    return tuple(
        ModelIdentityRef(
            profile_key=item.profile_key,
            provider=item.provider,
            model_name=item.model_name,
        )
        for item in outcomes
    )


def _source_refs(
    outcomes: tuple,
) -> tuple[OutcomeSourceRef, ...]:
    return tuple(
        OutcomeSourceRef(
            profile_key=item.profile_key,
            qualification_task_id=item.qualification_task_id,
            matrix_version=item.matrix_version,
            evaluated_at=item.evaluated_at,
        )
        for item in outcomes
    )


def _validate_request(
    request: CrossModelBehavioralAnalysisRequest,
) -> CrossModelAnalysisStatus | None:
    if not request.outcomes:
        return CrossModelAnalysisStatus.INSUFFICIENT_DATA
    profile_keys = [item.profile_key for item in request.outcomes]
    if len(profile_keys) != len(set(profile_keys)):
        return CrossModelAnalysisStatus.DUPLICATE_MODELS
    for outcome in request.outcomes:
        if outcome.matrix_version != request.matrix_version:
            return CrossModelAnalysisStatus.VERSION_MISMATCH
    return None


def run_cross_model_behavioral_analysis(
    request: CrossModelBehavioralAnalysisRequest,
    analyzers: tuple[BehaviorAnalyzer, ...] | None = None,
    *,
    analyzed_at: datetime | None = None,
) -> BehavioralComparisonResult:
    stamp = analyzed_at or datetime.now(tz=UTC)
    validation_status = _validate_request(request)
    active_analyzers = (
        analyzers if analyzers is not None else default_behavior_analyzers()
    )

    if validation_status is not None:
        return BehavioralComparisonResult(
            analysis_task_id=ANALYSIS_TASK_ID,
            scenario_id=request.scenario_id,
            matrix_version=request.matrix_version,
            analyzed_at=stamp,
            status=validation_status,
            models=_model_refs(request.outcomes),
            source_outcome_refs=_source_refs(request.outcomes),
            findings=(),
        )

    findings = tuple(
        analyzer.analyze(request.outcomes) for analyzer in active_analyzers
    )
    return BehavioralComparisonResult(
        analysis_task_id=ANALYSIS_TASK_ID,
        scenario_id=request.scenario_id,
        matrix_version=request.matrix_version,
        analyzed_at=stamp,
        status=CrossModelAnalysisStatus.COMPLETE,
        models=_model_refs(request.outcomes),
        source_outcome_refs=_source_refs(request.outcomes),
        findings=findings,
    )


__all__ = ["run_cross_model_behavioral_analysis"]
