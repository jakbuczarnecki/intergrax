# © Artur Czarnecki. All rights reserved.

"""Typed contracts for cross-model behavioral comparison (DS-E2E-15J-L2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    QualificationSessionState,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)

ANALYSIS_TASK_ID = "DS-E2E-15J-L2.CROSS-MODEL-BEHAVIORAL-ANALYSIS"


class ComparisonArea(StrEnum):
    COHORT_EXECUTION = "cohort_execution"
    QUALIFICATION_EXIT = "qualification_exit"
    SESSION_LIFECYCLE = "session_lifecycle"


class CrossModelAnalysisStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_DATA = "insufficient_data"
    VERSION_MISMATCH = "version_mismatch"
    DUPLICATE_MODELS = "duplicate_models"


class AreaAnalysisStatus(StrEnum):
    UNIFORM = "uniform"
    DIFFERS = "differs"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class CrossModelBehavioralAnalysisRequest:
    scenario_id: str
    matrix_version: str
    outcomes: tuple[ModelQualificationOutcome, ...]


@dataclass(frozen=True, slots=True)
class ModelIdentityRef:
    profile_key: str
    provider: str
    model_name: str


@dataclass(frozen=True, slots=True)
class OutcomeSourceRef:
    profile_key: str
    qualification_task_id: str
    matrix_version: str
    evaluated_at: datetime


@dataclass(frozen=True, slots=True)
class PairwiseBehavioralDifference:
    comparison_area: ComparisonArea
    analyzer_id: str
    profile_key_left: str
    profile_key_right: str
    left_descriptor: str
    right_descriptor: str


@dataclass(frozen=True, slots=True)
class CohortExecutionDifference(PairwiseBehavioralDifference):
    left_status: CohortExecutionStatus
    right_status: CohortExecutionStatus


@dataclass(frozen=True, slots=True)
class QualificationExitDifference(PairwiseBehavioralDifference):
    left_exit_code: QualificationCliExit
    right_exit_code: QualificationCliExit


@dataclass(frozen=True, slots=True)
class SessionLifecycleDifference(PairwiseBehavioralDifference):
    left_session_state: QualificationSessionState | None
    right_session_state: QualificationSessionState | None


@dataclass(frozen=True, slots=True)
class ComparisonAreaFinding:
    comparison_area: ComparisonArea
    analyzer_id: str
    status: AreaAnalysisStatus
    differences: tuple[PairwiseBehavioralDifference, ...]


@dataclass(frozen=True, slots=True)
class BehavioralComparisonResult:
    analysis_task_id: str
    scenario_id: str
    matrix_version: str
    analyzed_at: datetime
    status: CrossModelAnalysisStatus
    models: tuple[ModelIdentityRef, ...]
    source_outcome_refs: tuple[OutcomeSourceRef, ...]
    findings: tuple[ComparisonAreaFinding, ...]


__all__ = [
    "ANALYSIS_TASK_ID",
    "AreaAnalysisStatus",
    "BehavioralComparisonResult",
    "CohortExecutionDifference",
    "ComparisonArea",
    "ComparisonAreaFinding",
    "CrossModelAnalysisStatus",
    "CrossModelBehavioralAnalysisRequest",
    "ModelIdentityRef",
    "OutcomeSourceRef",
    "PairwiseBehavioralDifference",
    "QualificationExitDifference",
    "SessionLifecycleDifference",
]
