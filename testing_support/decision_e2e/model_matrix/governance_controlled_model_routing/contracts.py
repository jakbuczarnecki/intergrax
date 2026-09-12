# © Artur Czarnecki. All rights reserved.

"""Typed contracts for governance-controlled model routing (DS-E2E-15J-L5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from testing_support.decision_e2e.model_matrix.model_capability_baseline.contracts import (
    ModelCapabilityProfile,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    ModelEvidenceReference,
    ModelSelectionRecommendation,
)

GOVERNANCE_TASK_ID = "DS-E2E-15J-L5.GOVERNANCE-CONTROLLED-MODEL-ROUTING"
GOVERNANCE_VERSION = "1"


class GovernanceDisposition(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"


class GovernanceRiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DataSensitivityClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    FINANCIAL = "financial"
    CUSTOMER_PII = "customer_pii"
    CONFIDENTIAL = "confidential"


class GovernanceReasonCode(StrEnum):
    NO_MODEL_RECOMMENDATION = "no_model_recommendation"
    SELECTION_NOT_RECOMMENDED = "selection_not_recommended"
    NO_APPLICABLE_POLICY_EVALUATOR = "no_applicable_policy_evaluator"
    DATA_CLASSIFICATION_DENIED = "data_classification_denied"
    QUALIFICATION_EVIDENCE_MISSING = "qualification_evidence_missing"
    QUALIFICATION_SAFETY_LIMIT = "qualification_safety_limit"
    HIGH_RISK_REQUIRES_APPROVAL = "high_risk_requires_approval"
    POLICY_ALLOW = "policy_allow"


class GovernanceDataSourceKind(StrEnum):
    MODEL_SELECTION_RECOMMENDATION = "model_selection_recommendation"
    MODEL_CAPABILITY_PROFILE = "model_capability_profile"
    SELECTION_EVIDENCE_REFERENCE = "selection_evidence_reference"


@dataclass(frozen=True, slots=True)
class GovernancePolicyRef:
    policy_id: str
    policy_version: str


@dataclass(frozen=True, slots=True)
class GovernanceTaskContext:
    scenario_id: str
    data_sensitivity: DataSensitivityClass
    risk_tier: GovernanceRiskTier


@dataclass(frozen=True, slots=True)
class GovernanceEvaluationRequest:
    model_recommendation: ModelSelectionRecommendation | None
    task_context: GovernanceTaskContext
    applicable_policies: tuple[GovernancePolicyRef, ...]
    capability_evidence: tuple[ModelCapabilityProfile, ...]


@dataclass(frozen=True, slots=True)
class GovernanceReasonRef:
    reason_code: GovernanceReasonCode
    summary: str
    policy_id: str


@dataclass(frozen=True, slots=True)
class GovernanceDataSourceRef:
    source_kind: GovernanceDataSourceKind
    reference_id: str


@dataclass(frozen=True, slots=True)
class PolicyParticipationRecord:
    policy_id: str
    policy_version: str
    contribution: GovernanceDisposition
    outcome_summary: str


@dataclass(frozen=True, slots=True)
class GovernanceDecisionAuditMetadata:
    governance_task_id: str
    governance_version: str
    decision_id: str
    evaluated_at: datetime
    scenario_id: str
    recommended_profile_key: str | None
    applied_policy_refs: tuple[GovernancePolicyRef, ...]
    policy_participation: tuple[PolicyParticipationRecord, ...]
    data_source_refs: tuple[GovernanceDataSourceRef, ...]
    selection_evidence_refs: tuple[ModelEvidenceReference, ...]
    capability_profile_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    disposition: GovernanceDisposition
    reason_references: tuple[GovernanceReasonRef, ...]
    policy_references: tuple[GovernancePolicyRef, ...]
    audit_metadata: GovernanceDecisionAuditMetadata


__all__ = [
    "GOVERNANCE_TASK_ID",
    "GOVERNANCE_VERSION",
    "DataSensitivityClass",
    "GovernanceDataSourceKind",
    "GovernanceDataSourceRef",
    "GovernanceDecision",
    "GovernanceDecisionAuditMetadata",
    "GovernanceDisposition",
    "GovernanceEvaluationRequest",
    "GovernancePolicyRef",
    "GovernanceReasonCode",
    "GovernanceReasonRef",
    "GovernanceRiskTier",
    "GovernanceTaskContext",
    "ModelCapabilityProfile",
    "ModelSelectionRecommendation",
    "PolicyParticipationRecord",
]
