# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline import (
    CapabilityDimensionId,
    CapabilityProfileBuildRequest,
    ObservationLevel,
    build_model_capability_profiles,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation import (
    CapabilityMatchStrategy,
    CapabilitySelectionConstraints,
    CostPreferenceStrategy,
    ModelSelectionEngine,
    ModelSelectionRequest,
    ModelSelectionStatus,
    SafetyLimitationStrategy,
    SELECTION_TASK_ID,
    TaskCapabilityRequirement,
    TaskRequirements,
    default_selection_strategies,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.registry import (
    qualification_matrix_version,
)


def _outcome(
    profile_key: str,
    *,
    exit_code: QualificationCliExit = QualificationCliExit.SUCCESS,
) -> ModelQualificationOutcome:
    stamp = datetime(2026, 9, 12, 9, 0, 0, tzinfo=UTC)
    return ModelQualificationOutcome(
        profile_key=profile_key,
        provider="ollama",
        model_name=profile_key,
        matrix_version=qualification_matrix_version(),
        qualification_task_id="DS-E2E-15J-L1.R6",
        evaluated_at=stamp,
        status=CohortExecutionStatus.EXECUTED,
        exit_code=exit_code,
        session_state=QualificationSessionState.FINALIZED,
    )


def _profiles(
    *profile_keys: str, exit_codes: dict[str, QualificationCliExit] | None = None
):
    matrix_version = qualification_matrix_version()
    codes = exit_codes or {}
    outcomes = tuple(
        _outcome(key, exit_code=codes.get(key, QualificationCliExit.SUCCESS))
        for key in profile_keys
    )
    built = build_model_capability_profiles(
        CapabilityProfileBuildRequest(matrix_version=matrix_version, outcomes=outcomes)
    )
    return built.profiles


def _request(
    profiles: tuple,
    *,
    minimum: ObservationLevel = ObservationLevel.MODERATE,
) -> ModelSelectionRequest:
    return ModelSelectionRequest(
        task_requirements=TaskRequirements(
            scenario_id="selection-basic",
            capability_requirements=(
                TaskCapabilityRequirement(
                    dimension_id=CapabilityDimensionId.QUALIFICATION_EXIT,
                    minimum_level=minimum,
                ),
            ),
        ),
        capability_constraints=CapabilitySelectionConstraints(
            required_matrix_version=qualification_matrix_version(),
            excluded_profile_keys=(),
            require_behavioral_baseline=False,
        ),
        available_model_profiles=profiles,
    )


def test_basic_recommendation_two_profiles() -> None:
    profiles = _profiles(
        "model-a",
        "model-b",
        exit_codes={"model-b": QualificationCliExit.CRITICAL_SAFETY_FAILURE},
    )
    engine = ModelSelectionEngine(strategies=default_selection_strategies())
    recommendation = engine.recommend(
        _request(profiles),
        recommended_at=datetime(2026, 9, 12, 11, 0, 0, tzinfo=UTC),
    )

    assert recommendation.status is ModelSelectionStatus.RECOMMENDED
    assert recommendation.selected_model_reference is not None
    assert recommendation.selected_model_reference.profile_key == "model-a"
    assert recommendation.decision_metadata.selection_task_id == SELECTION_TASK_ID
    assert set(recommendation.decision_metadata.analyzed_profile_keys) == {
        "model-a",
        "model-b",
    }


def test_engine_uses_all_injected_strategies() -> None:
    profiles = _profiles("model-a", "model-b")
    strategies = (
        CapabilityMatchStrategy(),
        SafetyLimitationStrategy(),
        CostPreferenceStrategy(),
    )
    engine = ModelSelectionEngine(strategies=strategies)
    recommendation = engine.recommend(_request(profiles))

    assert recommendation.decision_metadata.strategy_ids == (
        "capability_match",
        "safety_limitation",
        "cost_preference",
    )
    assert len(recommendation.decision_metadata.strategy_participation) == 3


def test_custom_strategy_plugs_in_without_engine_edit() -> None:
    class _PreferModelBStrategy:
        strategy_id = "prefer_model_b"

        def evaluate(self, request: ModelSelectionRequest):
            from testing_support.decision_e2e.model_matrix.model_selection_recommendation.protocol import (
                SelectionStrategyResult,
                StrategyModelAssessment,
            )

            assessments = tuple(
                StrategyModelAssessment(
                    profile_key=profile.model_identity.profile_key,
                    eligible=True,
                    preference_rank=0
                    if profile.model_identity.profile_key == "model-b"
                    else 1,
                    matched_capabilities=(),
                    unmet_requirements=(),
                    outcome_summary="test plugin prefers model-b",
                )
                for profile in request.available_model_profiles
            )
            return SelectionStrategyResult(
                strategy_id=self.strategy_id,
                assessments=assessments,
            )

    profiles = _profiles("model-a", "model-b")
    engine = ModelSelectionEngine(
        strategies=(
            CapabilityMatchStrategy(),
            SafetyLimitationStrategy(),
            _PreferModelBStrategy(),
        )
    )
    recommendation = engine.recommend(_request(profiles, minimum=ObservationLevel.WEAK))

    assert recommendation.status is ModelSelectionStatus.RECOMMENDED
    assert recommendation.selected_model_reference is not None
    assert recommendation.selected_model_reference.profile_key == "model-b"
    assert "prefer_model_b" in recommendation.decision_metadata.strategy_ids


def test_no_suitable_model_when_requirements_unmet() -> None:
    profiles = _profiles(
        "model-a",
        "model-b",
        exit_codes={
            "model-a": QualificationCliExit.CRITICAL_SAFETY_FAILURE,
            "model-b": QualificationCliExit.CRITICAL_SAFETY_FAILURE,
        },
    )
    engine = ModelSelectionEngine(strategies=default_selection_strategies())
    recommendation = engine.recommend(
        _request(profiles, minimum=ObservationLevel.STRONG)
    )

    assert recommendation.status is ModelSelectionStatus.NO_SUITABLE_MODEL
    assert recommendation.selected_model_reference is None
    assert recommendation.unmet_requirements
