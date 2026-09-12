# © Artur Czarnecki. All rights reserved.

"""Typed contracts for model selection recommendation (DS-E2E-15J-L4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    ModelIdentityRef,
    OutcomeSourceRef,
)
from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
    CapabilityDimensionId,
    ModelCapabilityProfile,
    ObservationLevel,
)

SELECTION_TASK_ID = "DS-E2E-15J-L4.INTELLIGENT-MODEL-SELECTION-LAYER"
SELECTION_VERSION = "1"


class ModelSelectionStatus(StrEnum):
    RECOMMENDED = "recommended"
    NO_SUITABLE_MODEL = "no_suitable_model"
    INSUFFICIENT_INPUT = "insufficient_input"


@dataclass(frozen=True, slots=True)
class TaskCapabilityRequirement:
    dimension_id: CapabilityDimensionId
    minimum_level: ObservationLevel


@dataclass(frozen=True, slots=True)
class TaskRequirements:
    scenario_id: str
    capability_requirements: tuple[TaskCapabilityRequirement, ...]


@dataclass(frozen=True, slots=True)
class CapabilitySelectionConstraints:
    required_matrix_version: str | None
    excluded_profile_keys: tuple[str, ...]
    require_behavioral_baseline: bool


@dataclass(frozen=True, slots=True)
class ModelSelectionRequest:
    task_requirements: TaskRequirements
    capability_constraints: CapabilitySelectionConstraints
    available_model_profiles: tuple[ModelCapabilityProfile, ...]


@dataclass(frozen=True, slots=True)
class MatchedCapabilityEvidence:
    dimension_id: CapabilityDimensionId
    observation_level: ObservationLevel
    factual_descriptor: str
    extractor_id: str


@dataclass(frozen=True, slots=True)
class UnmetRequirementEvidence:
    dimension_id: CapabilityDimensionId
    required_minimum: ObservationLevel
    best_observed_level: ObservationLevel | None
    reason: str


@dataclass(frozen=True, slots=True)
class ModelEvidenceReference:
    profile_key: str
    baseline_task_id: str
    baseline_version: str
    qualification_source_refs: tuple[OutcomeSourceRef, ...]
    behavioral_analysis_task_id: str | None


@dataclass(frozen=True, slots=True)
class StrategyParticipationRecord:
    strategy_id: str
    outcome_summary: str


@dataclass(frozen=True, slots=True)
class ModelSelectionDecisionMetadata:
    selection_task_id: str
    selection_version: str
    recommended_at: datetime
    scenario_id: str
    analyzed_profile_keys: tuple[str, ...]
    task_requirement_count: int
    strategy_ids: tuple[str, ...]
    strategy_participation: tuple[StrategyParticipationRecord, ...]


@dataclass(frozen=True, slots=True)
class ModelSelectionRecommendation:
    status: ModelSelectionStatus
    selected_model_reference: ModelIdentityRef | None
    matched_capabilities: tuple[MatchedCapabilityEvidence, ...]
    unmet_requirements: tuple[UnmetRequirementEvidence, ...]
    evidence_references: tuple[ModelEvidenceReference, ...]
    decision_metadata: ModelSelectionDecisionMetadata


def observation_level_rank(level: ObservationLevel) -> int:
    order = (
        ObservationLevel.UNKNOWN,
        ObservationLevel.WEAK,
        ObservationLevel.NEUTRAL,
        ObservationLevel.MODERATE,
        ObservationLevel.STRONG,
    )
    return order.index(level)


def observation_meets_minimum(
    observed: ObservationLevel,
    minimum: ObservationLevel,
) -> bool:
    return observation_level_rank(observed) >= observation_level_rank(minimum)


__all__ = [
    "CapabilitySelectionConstraints",
    "MatchedCapabilityEvidence",
    "ModelEvidenceReference",
    "ModelSelectionDecisionMetadata",
    "ModelSelectionRecommendation",
    "ModelSelectionRequest",
    "ModelSelectionStatus",
    "SELECTION_TASK_ID",
    "SELECTION_VERSION",
    "StrategyParticipationRecord",
    "TaskCapabilityRequirement",
    "TaskRequirements",
    "UnmetRequirementEvidence",
    "observation_level_rank",
    "observation_meets_minimum",
]
