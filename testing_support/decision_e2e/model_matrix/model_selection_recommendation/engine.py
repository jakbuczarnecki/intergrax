# © Artur Czarnecki. All rights reserved.

"""Model selection recommendation orchestration (DS-E2E-15J-L4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    ModelIdentityRef,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    SELECTION_TASK_ID,
    SELECTION_VERSION,
    MatchedCapabilityEvidence,
    ModelEvidenceReference,
    ModelSelectionDecisionMetadata,
    ModelSelectionRecommendation,
    ModelSelectionRequest,
    ModelSelectionStatus,
    StrategyParticipationRecord,
    UnmetRequirementEvidence,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.protocol import (
    SelectionStrategy,
    SelectionStrategyResult,
    StrategyModelAssessment,
)


def _profile_evidence(profile) -> ModelEvidenceReference:
    behavioral_task_id = (
        profile.behavioral_source_ref.analysis_task_id
        if profile.behavioral_source_ref is not None
        else None
    )
    return ModelEvidenceReference(
        profile_key=profile.model_identity.profile_key,
        baseline_task_id=profile.baseline_task_id,
        baseline_version=profile.baseline_version,
        qualification_source_refs=profile.qualification_source_refs,
        behavioral_analysis_task_id=behavioral_task_id,
    )


def _filter_profiles(request: ModelSelectionRequest):
    constraints = request.capability_constraints
    filtered = []
    for profile in request.available_model_profiles:
        key = profile.model_identity.profile_key
        if key in constraints.excluded_profile_keys:
            continue
        if (
            constraints.required_matrix_version is not None
            and profile.matrix_version != constraints.required_matrix_version
        ):
            continue
        if (
            constraints.require_behavioral_baseline
            and profile.behavioral_source_ref is None
        ):
            continue
        filtered.append(profile)
    return tuple(filtered)


def _assessment_for_key(
    assessments: tuple[StrategyModelAssessment, ...],
    profile_key: str,
) -> StrategyModelAssessment | None:
    for item in assessments:
        if item.profile_key == profile_key:
            return item
    return None


def _merge_capability_evidence(
    chunks: tuple[tuple[MatchedCapabilityEvidence, ...], ...],
) -> tuple[MatchedCapabilityEvidence, ...]:
    merged: list[MatchedCapabilityEvidence] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        for item in chunk:
            key = (item.dimension_id.value, item.extractor_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return tuple(merged)


def _merge_unmet(
    chunks: tuple[tuple[UnmetRequirementEvidence, ...], ...],
) -> tuple[UnmetRequirementEvidence, ...]:
    return tuple(item for chunk in chunks for item in chunk)


@dataclass(frozen=True, slots=True)
class ModelSelectionEngine:
    """Produces auditable model recommendations; does not execute models."""

    strategies: tuple[SelectionStrategy, ...]

    def recommend(
        self,
        request: ModelSelectionRequest,
        *,
        recommended_at: datetime | None = None,
    ) -> ModelSelectionRecommendation:
        stamp = recommended_at or datetime.now(tz=UTC)
        strategy_ids = tuple(strategy.strategy_id for strategy in self.strategies)

        if not request.available_model_profiles:
            return ModelSelectionRecommendation(
                status=ModelSelectionStatus.INSUFFICIENT_INPUT,
                selected_model_reference=None,
                matched_capabilities=(),
                unmet_requirements=(),
                evidence_references=(),
                decision_metadata=ModelSelectionDecisionMetadata(
                    selection_task_id=SELECTION_TASK_ID,
                    selection_version=SELECTION_VERSION,
                    recommended_at=stamp,
                    scenario_id=request.task_requirements.scenario_id,
                    analyzed_profile_keys=(),
                    task_requirement_count=len(
                        request.task_requirements.capability_requirements
                    ),
                    strategy_ids=strategy_ids,
                    strategy_participation=(),
                ),
            )

        if not self.strategies:
            return ModelSelectionRecommendation(
                status=ModelSelectionStatus.INSUFFICIENT_INPUT,
                selected_model_reference=None,
                matched_capabilities=(),
                unmet_requirements=(),
                evidence_references=tuple(
                    _profile_evidence(profile)
                    for profile in request.available_model_profiles
                ),
                decision_metadata=ModelSelectionDecisionMetadata(
                    selection_task_id=SELECTION_TASK_ID,
                    selection_version=SELECTION_VERSION,
                    recommended_at=stamp,
                    scenario_id=request.task_requirements.scenario_id,
                    analyzed_profile_keys=tuple(
                        item.model_identity.profile_key
                        for item in request.available_model_profiles
                    ),
                    task_requirement_count=len(
                        request.task_requirements.capability_requirements
                    ),
                    strategy_ids=(),
                    strategy_participation=(),
                ),
            )

        filtered_profiles = _filter_profiles(request)
        filtered_request = ModelSelectionRequest(
            task_requirements=request.task_requirements,
            capability_constraints=request.capability_constraints,
            available_model_profiles=filtered_profiles,
        )
        analyzed_keys = tuple(
            item.model_identity.profile_key for item in filtered_profiles
        )
        evidence_refs = tuple(
            _profile_evidence(profile) for profile in filtered_profiles
        )

        strategy_results: tuple[SelectionStrategyResult, ...] = tuple(
            strategy.evaluate(filtered_request) for strategy in self.strategies
        )
        participation = tuple(
            StrategyParticipationRecord(
                strategy_id=result.strategy_id,
                outcome_summary=_participation_summary(result),
            )
            for result in strategy_results
        )

        if not filtered_profiles:
            return ModelSelectionRecommendation(
                status=ModelSelectionStatus.NO_SUITABLE_MODEL,
                selected_model_reference=None,
                matched_capabilities=(),
                unmet_requirements=(),
                evidence_references=evidence_refs,
                decision_metadata=ModelSelectionDecisionMetadata(
                    selection_task_id=SELECTION_TASK_ID,
                    selection_version=SELECTION_VERSION,
                    recommended_at=stamp,
                    scenario_id=request.task_requirements.scenario_id,
                    analyzed_profile_keys=analyzed_keys,
                    task_requirement_count=len(
                        request.task_requirements.capability_requirements
                    ),
                    strategy_ids=strategy_ids,
                    strategy_participation=participation,
                ),
            )

        candidates: list[
            tuple[
                str,
                int,
                tuple[MatchedCapabilityEvidence, ...],
                tuple[UnmetRequirementEvidence, ...],
            ]
        ] = []
        for profile in filtered_profiles:
            key = profile.model_identity.profile_key
            per_strategy = tuple(
                _assessment_for_key(result.assessments, key)
                for result in strategy_results
            )
            if any(item is None for item in per_strategy):
                continue
            eligible = all(item.eligible for item in per_strategy if item is not None)
            if not eligible:
                continue
            rank = sum(
                item.preference_rank for item in per_strategy if item is not None
            )
            matched = _merge_capability_evidence(
                tuple(
                    item.matched_capabilities
                    for item in per_strategy
                    if item is not None
                )
            )
            unmet = _merge_unmet(
                tuple(
                    item.unmet_requirements for item in per_strategy if item is not None
                )
            )
            candidates.append((key, rank, matched, unmet))

        if not candidates:
            all_unmet = _merge_unmet(
                tuple(
                    assessment.unmet_requirements
                    for result in strategy_results
                    for assessment in result.assessments
                )
            )
            return ModelSelectionRecommendation(
                status=ModelSelectionStatus.NO_SUITABLE_MODEL,
                selected_model_reference=None,
                matched_capabilities=(),
                unmet_requirements=all_unmet,
                evidence_references=evidence_refs,
                decision_metadata=ModelSelectionDecisionMetadata(
                    selection_task_id=SELECTION_TASK_ID,
                    selection_version=SELECTION_VERSION,
                    recommended_at=stamp,
                    scenario_id=request.task_requirements.scenario_id,
                    analyzed_profile_keys=analyzed_keys,
                    task_requirement_count=len(
                        request.task_requirements.capability_requirements
                    ),
                    strategy_ids=strategy_ids,
                    strategy_participation=participation,
                ),
            )

        chosen_key, _, chosen_matched, chosen_unmet = min(
            candidates,
            key=lambda item: (item[1], item[0]),
        )
        chosen_profile = next(
            item
            for item in filtered_profiles
            if item.model_identity.profile_key == chosen_key
        )
        identity = ModelIdentityRef(
            profile_key=chosen_profile.model_identity.profile_key,
            provider=chosen_profile.model_identity.provider,
            model_name=chosen_profile.model_identity.model_name,
        )
        return ModelSelectionRecommendation(
            status=ModelSelectionStatus.RECOMMENDED,
            selected_model_reference=identity,
            matched_capabilities=chosen_matched,
            unmet_requirements=chosen_unmet,
            evidence_references=evidence_refs,
            decision_metadata=ModelSelectionDecisionMetadata(
                selection_task_id=SELECTION_TASK_ID,
                selection_version=SELECTION_VERSION,
                recommended_at=stamp,
                scenario_id=request.task_requirements.scenario_id,
                analyzed_profile_keys=analyzed_keys,
                task_requirement_count=len(
                    request.task_requirements.capability_requirements
                ),
                strategy_ids=strategy_ids,
                strategy_participation=participation,
            ),
        )


def _participation_summary(result: SelectionStrategyResult) -> str:
    eligible_count = sum(1 for item in result.assessments if item.eligible)
    return f"{result.strategy_id}: {eligible_count}/{len(result.assessments)} eligible"


__all__ = ["ModelSelectionEngine"]
