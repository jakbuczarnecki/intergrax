# © Artur Czarnecki. All rights reserved.

"""Typed contracts for model capability baseline profiles (DS-E2E-15J-L3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.cross_model_behavioral_analysis.contracts import (
    BehavioralComparisonResult,
    ModelIdentityRef,
    OutcomeSourceRef,
)
from testing_support.decision_e2e.model_matrix.model_qualification_outcome import (
    ModelQualificationOutcome,
)

BASELINE_TASK_ID = "DS-E2E-15J-L3.MODEL-CAPABILITY-BASELINE"
BASELINE_VERSION = "1"


class CapabilityProfileBuildStatus(StrEnum):
    COMPLETE = "complete"
    INSUFFICIENT_DATA = "insufficient_data"
    VERSION_MISMATCH = "version_mismatch"


class CapabilityDimensionId(StrEnum):
    QUALIFICATION_EXIT = "qualification_exit"
    COHORT_EXECUTION = "cohort_execution"
    SESSION_LIFECYCLE = "session_lifecycle"
    BEHAVIORAL_PEER_ALIGNMENT = "behavioral_peer_alignment"


class ObservationLevel(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BehavioralAnalysisSourceRef:
    analysis_task_id: str
    scenario_id: str
    matrix_version: str
    analyzed_at: datetime


@dataclass(frozen=True, slots=True)
class CapabilityProfileBuildRequest:
    matrix_version: str
    outcomes: tuple[ModelQualificationOutcome, ...]
    behavioral_comparison: BehavioralComparisonResult | None = None


@dataclass(frozen=True, slots=True)
class CapabilityObservation:
    dimension_id: CapabilityDimensionId
    extractor_id: str
    level: ObservationLevel
    factual_descriptor: str
    qualification_evidence: tuple[OutcomeSourceRef, ...]
    behavioral_evidence: tuple[BehavioralAnalysisSourceRef, ...]


@dataclass(frozen=True, slots=True)
class LimitationObservation:
    dimension_id: CapabilityDimensionId
    extractor_id: str
    level: ObservationLevel
    factual_descriptor: str
    qualification_evidence: tuple[OutcomeSourceRef, ...]
    behavioral_evidence: tuple[BehavioralAnalysisSourceRef, ...]


@dataclass(frozen=True, slots=True)
class ModelCapabilityProfile:
    model_identity: ModelIdentityRef
    model_version: str
    matrix_version: str
    baseline_task_id: str
    baseline_version: str
    generated_at: datetime
    qualification_source_refs: tuple[OutcomeSourceRef, ...]
    behavioral_source_ref: BehavioralAnalysisSourceRef | None
    source_qualification_versions: tuple[str, ...]
    source_behavioral_matrix_version: str | None
    capabilities: tuple[CapabilityObservation, ...]
    limitations: tuple[LimitationObservation, ...]


@dataclass(frozen=True, slots=True)
class CapabilityProfileBuildResult:
    build_task_id: str
    baseline_version: str
    matrix_version: str
    generated_at: datetime
    status: CapabilityProfileBuildStatus
    profiles: tuple[ModelCapabilityProfile, ...]


__all__ = [
    "BASELINE_TASK_ID",
    "BASELINE_VERSION",
    "BehavioralAnalysisSourceRef",
    "CapabilityDimensionId",
    "CapabilityObservation",
    "CapabilityProfileBuildRequest",
    "CapabilityProfileBuildResult",
    "CapabilityProfileBuildStatus",
    "LimitationObservation",
    "ModelCapabilityProfile",
    "ObservationLevel",
]
