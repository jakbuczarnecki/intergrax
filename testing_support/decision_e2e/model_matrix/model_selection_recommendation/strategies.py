# © Artur Czarnecki. All rights reserved.

"""Default selection strategies (DS-E2E-15J-L4)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
    CapabilityDimensionId,
    CapabilityObservation,
    LimitationObservation,
    ObservationLevel,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    MatchedCapabilityEvidence,
    ModelSelectionRequest,
    UnmetRequirementEvidence,
    observation_level_rank,
    observation_meets_minimum,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.protocol import (
    SelectionStrategyResult,
    StrategyModelAssessment,
)

_CAPABILITY_MATCH_STRATEGY_ID = "capability_match"
_SAFETY_LIMITATION_STRATEGY_ID = "safety_limitation"
_COST_PREFERENCE_STRATEGY_ID = "cost_preference"

_SAFETY_DIMENSIONS = (
    CapabilityDimensionId.QUALIFICATION_EXIT,
    CapabilityDimensionId.SESSION_LIFECYCLE,
)


def _best_capability_level(
    profile_key: str,
    dimension_id: CapabilityDimensionId,
    capabilities: tuple[CapabilityObservation, ...],
    limitations: tuple[LimitationObservation, ...],
) -> tuple[
    ObservationLevel | None, CapabilityObservation | LimitationObservation | None
]:
    _ = profile_key
    cap_matches = [item for item in capabilities if item.dimension_id is dimension_id]
    if cap_matches:
        best = max(cap_matches, key=lambda item: observation_level_rank(item.level))
        return best.level, best
    lim_matches = [item for item in limitations if item.dimension_id is dimension_id]
    if lim_matches:
        worst = min(lim_matches, key=lambda item: observation_level_rank(item.level))
        return worst.level, worst
    return None, None


class CapabilityMatchStrategy:
    """Require task capability dimensions to meet minimum observation levels."""

    @property
    def strategy_id(self) -> str:
        return _CAPABILITY_MATCH_STRATEGY_ID

    def evaluate(self, request: ModelSelectionRequest) -> SelectionStrategyResult:
        requirements = request.task_requirements.capability_requirements
        assessments: list[StrategyModelAssessment] = []
        for profile in request.available_model_profiles:
            key = profile.model_identity.profile_key
            matched: list[MatchedCapabilityEvidence] = []
            unmet: list[UnmetRequirementEvidence] = []
            for requirement in requirements:
                level, source = _best_capability_level(
                    key,
                    requirement.dimension_id,
                    profile.capabilities,
                    profile.limitations,
                )
                if level is None:
                    unmet.append(
                        UnmetRequirementEvidence(
                            dimension_id=requirement.dimension_id,
                            required_minimum=requirement.minimum_level,
                            best_observed_level=None,
                            reason="no capability observation for dimension",
                        )
                    )
                    continue
                if observation_meets_minimum(level, requirement.minimum_level):
                    descriptor = (
                        source.factual_descriptor
                        if source is not None
                        else "requirement satisfied"
                    )
                    extractor_id = (
                        source.extractor_id if source is not None else "unknown"
                    )
                    matched.append(
                        MatchedCapabilityEvidence(
                            dimension_id=requirement.dimension_id,
                            observation_level=level,
                            factual_descriptor=descriptor,
                            extractor_id=extractor_id,
                        )
                    )
                else:
                    unmet.append(
                        UnmetRequirementEvidence(
                            dimension_id=requirement.dimension_id,
                            required_minimum=requirement.minimum_level,
                            best_observed_level=level,
                            reason="observed level below task minimum",
                        )
                    )
            eligible = not unmet
            summary = (
                f"all {len(requirements)} requirements met"
                if eligible
                else f"{len(unmet)} unmet of {len(requirements)} requirements"
            )
            assessments.append(
                StrategyModelAssessment(
                    profile_key=key,
                    eligible=eligible,
                    preference_rank=0,
                    matched_capabilities=tuple(matched),
                    unmet_requirements=tuple(unmet),
                    outcome_summary=summary,
                )
            )
        return SelectionStrategyResult(
            strategy_id=self.strategy_id,
            assessments=tuple(assessments),
        )


class SafetyLimitationStrategy:
    """Exclude models with weak safety-related limitations on baseline profile."""

    @property
    def strategy_id(self) -> str:
        return _SAFETY_LIMITATION_STRATEGY_ID

    def evaluate(self, request: ModelSelectionRequest) -> SelectionStrategyResult:
        assessments: list[StrategyModelAssessment] = []
        for profile in request.available_model_profiles:
            key = profile.model_identity.profile_key
            blocking = [
                item
                for item in profile.limitations
                if item.dimension_id in _SAFETY_DIMENSIONS
                and item.level in (ObservationLevel.WEAK, ObservationLevel.UNKNOWN)
            ]
            eligible = not blocking
            summary = (
                "no blocking safety limitations"
                if eligible
                else f"{len(blocking)} blocking safety limitation(s)"
            )
            assessments.append(
                StrategyModelAssessment(
                    profile_key=key,
                    eligible=eligible,
                    preference_rank=0,
                    matched_capabilities=(),
                    unmet_requirements=(),
                    outcome_summary=summary,
                )
            )
        return SelectionStrategyResult(
            strategy_id=self.strategy_id,
            assessments=tuple(assessments),
        )


class CostPreferenceStrategy:
    """Prefer models with fewer recorded limitations (factual baseline proxy)."""

    @property
    def strategy_id(self) -> str:
        return _COST_PREFERENCE_STRATEGY_ID

    def evaluate(self, request: ModelSelectionRequest) -> SelectionStrategyResult:
        assessments: list[StrategyModelAssessment] = []
        for profile in request.available_model_profiles:
            key = profile.model_identity.profile_key
            limitation_count = len(profile.limitations)
            assessments.append(
                StrategyModelAssessment(
                    profile_key=key,
                    eligible=True,
                    preference_rank=limitation_count,
                    matched_capabilities=(),
                    unmet_requirements=(),
                    outcome_summary=(
                        f"preference rank {limitation_count} from limitation count"
                    ),
                )
            )
        return SelectionStrategyResult(
            strategy_id=self.strategy_id,
            assessments=tuple(assessments),
        )


def default_selection_strategies() -> tuple[
    CapabilityMatchStrategy,
    SafetyLimitationStrategy,
    CostPreferenceStrategy,
]:
    return (
        CapabilityMatchStrategy(),
        SafetyLimitationStrategy(),
        CostPreferenceStrategy(),
    )


__all__ = [
    "CapabilityMatchStrategy",
    "CostPreferenceStrategy",
    "SafetyLimitationStrategy",
    "default_selection_strategies",
]
