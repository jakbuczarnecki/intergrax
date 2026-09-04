# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical deterministic worker obstacle classifier (AW-6A).

Classification order:
1. policy/authority
2. credential
3. explicit human/business gate
4. known dependency/reliability signals
5. schema/API drift
6. missing capability
7. unknown
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from intergrax.contracts.autonomous_work.obstacle_recovery import (
    ObstacleClassificationReasonCode,
    WorkerObstacleClassification,
    WorkerObstacleEvidence,
    WorkerObstacleKind,
    WorkerObstacleSourceKind,
    derive_worker_obstacle_id,
)
from intergrax.contracts.policy_action import PolicyAction
from intergrax.contracts.resilience_policy import FailureClass

_CLASSIFIER_ID: Final[str] = "canonical.deterministic.v1"

_TRANSIENT_RUNTIME_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "timeout",
        "runtime_error",
        "internal_error",
        "llm_error",
        "tool_error",
    }
)


class CanonicalWorkerObstacleClassifier:
    """Deterministic-first canonical obstacle classifier."""

    classifier_id: str = _CLASSIFIER_ID

    def classify(
        self,
        evidence: WorkerObstacleEvidence,
        *,
        classified_at: datetime,
    ) -> WorkerObstacleClassification:
        obstacle_id = derive_worker_obstacle_id(evidence)
        evidence_refs = evidence.problem_evidence_refs

        policy_kind, policy_reason = _classify_policy(evidence)
        if policy_kind is not None and policy_reason is not None:
            return _build_classification(
                obstacle_id=obstacle_id,
                obstacle_kind=policy_kind,
                reason_code=policy_reason,
                evidence_refs=evidence_refs,
                classified_at=classified_at,
            )

        credential_kind, credential_reason = _classify_credential(evidence)
        if credential_kind is not None and credential_reason is not None:
            return _build_classification(
                obstacle_id=obstacle_id,
                obstacle_kind=credential_kind,
                reason_code=credential_reason,
                evidence_refs=evidence_refs,
                classified_at=classified_at,
            )

        human_kind, human_reason = _classify_human_business(evidence)
        if human_kind is not None and human_reason is not None:
            return _build_classification(
                obstacle_id=obstacle_id,
                obstacle_kind=human_kind,
                reason_code=human_reason,
                evidence_refs=evidence_refs,
                classified_at=classified_at,
            )

        if evidence.suspicious_or_unsafe:
            return _build_classification(
                obstacle_id=obstacle_id,
                obstacle_kind=WorkerObstacleKind.SUSPICIOUS_OR_UNSAFE,
                reason_code=ObstacleClassificationReasonCode.SUSPICIOUS_OR_UNSAFE,
                evidence_refs=evidence_refs,
                classified_at=classified_at,
            )

        reliability_kind, reliability_reason = _classify_reliability(evidence)
        if reliability_kind is not None and reliability_reason is not None:
            return _build_classification(
                obstacle_id=obstacle_id,
                obstacle_kind=reliability_kind,
                reason_code=reliability_reason,
                evidence_refs=evidence_refs,
                classified_at=classified_at,
            )

        if evidence.schema_drift_detected:
            return _build_classification(
                obstacle_id=obstacle_id,
                obstacle_kind=WorkerObstacleKind.SCHEMA_OR_API_DRIFT,
                reason_code=ObstacleClassificationReasonCode.SCHEMA_OR_API_DRIFT,
                evidence_refs=evidence_refs,
                classified_at=classified_at,
            )

        if (
            evidence.capability_missing_ref is not None
            or evidence.source_kind is WorkerObstacleSourceKind.CAPABILITY_RESOLUTION
        ):
            return _build_classification(
                obstacle_id=obstacle_id,
                obstacle_kind=WorkerObstacleKind.CAPABILITY_MISSING,
                reason_code=ObstacleClassificationReasonCode.CAPABILITY_MISSING,
                evidence_refs=evidence_refs,
                classified_at=classified_at,
            )

        if (
            evidence.alternative_path_ref is not None
            and evidence.source_kind
            in {
                WorkerObstacleSourceKind.EXECUTION_FAILURE,
                WorkerObstacleSourceKind.CAPABILITY_RESOLUTION,
                WorkerObstacleSourceKind.OPERATOR,
            }
        ):
            return _build_classification(
                obstacle_id=obstacle_id,
                obstacle_kind=WorkerObstacleKind.ALTERNATIVE_PATH_AVAILABLE,
                reason_code=ObstacleClassificationReasonCode.ALTERNATIVE_PATH_AVAILABLE,
                evidence_refs=evidence_refs,
                classified_at=classified_at,
            )

        return _build_classification(
            obstacle_id=obstacle_id,
            obstacle_kind=WorkerObstacleKind.UNKNOWN,
            reason_code=ObstacleClassificationReasonCode.UNKNOWN_EVIDENCE,
            evidence_refs=evidence_refs,
            classified_at=classified_at,
        )


def _build_classification(
    *,
    obstacle_id: str,
    obstacle_kind: WorkerObstacleKind,
    reason_code: ObstacleClassificationReasonCode,
    evidence_refs: tuple,
    classified_at: datetime,
) -> WorkerObstacleClassification:
    return WorkerObstacleClassification(
        obstacle_id=obstacle_id,
        obstacle_kind=obstacle_kind,
        classifier_id=_CLASSIFIER_ID,
        reason_code=reason_code,
        evidence_refs=evidence_refs,
        classified_at=classified_at,
    )


def _classify_policy(
    evidence: WorkerObstacleEvidence,
) -> tuple[WorkerObstacleKind | None, ObstacleClassificationReasonCode | None]:
    decision = evidence.policy_decision
    if decision is None:
        return None, None
    action = decision.action
    if action is PolicyAction.DENY:
        return WorkerObstacleKind.POLICY_DENIED, ObstacleClassificationReasonCode.POLICY_DENIED
    if action is PolicyAction.REQUIRE_HUMAN:
        return (
            WorkerObstacleKind.HUMAN_DECISION_REQUIRED,
            ObstacleClassificationReasonCode.POLICY_REQUIRE_HUMAN,
        )
    if action is PolicyAction.ESCALATE:
        return (
            WorkerObstacleKind.HUMAN_DECISION_REQUIRED,
            ObstacleClassificationReasonCode.POLICY_ESCALATE,
        )
    if action is PolicyAction.MODIFY:
        if decision.modified_decision is None:
            return (
                WorkerObstacleKind.HUMAN_DECISION_REQUIRED,
                ObstacleClassificationReasonCode.POLICY_MODIFY_UNRESOLVED,
            )
    return None, None


def _classify_credential(
    evidence: WorkerObstacleEvidence,
) -> tuple[WorkerObstacleKind | None, ObstacleClassificationReasonCode | None]:
    if evidence.credential_ref is not None:
        return (
            WorkerObstacleKind.CREDENTIAL_UNAVAILABLE,
            ObstacleClassificationReasonCode.CREDENTIAL_UNAVAILABLE,
        )
    if evidence.runtime_error_code == "permission_error":
        return (
            WorkerObstacleKind.CREDENTIAL_UNAVAILABLE,
            ObstacleClassificationReasonCode.CREDENTIAL_UNAVAILABLE,
        )
    if evidence.failure_class is FailureClass.POLICY_ERROR and evidence.credential_ref:
        return (
            WorkerObstacleKind.CREDENTIAL_UNAVAILABLE,
            ObstacleClassificationReasonCode.CREDENTIAL_UNAVAILABLE,
        )
    return None, None


def _classify_human_business(
    evidence: WorkerObstacleEvidence,
) -> tuple[WorkerObstacleKind | None, ObstacleClassificationReasonCode | None]:
    if evidence.source_kind is WorkerObstacleSourceKind.HUMAN_GATE:
        return (
            WorkerObstacleKind.HUMAN_DECISION_REQUIRED,
            ObstacleClassificationReasonCode.HUMAN_GATE_PENDING,
        )
    if evidence.business_ambiguity or evidence.source_kind is WorkerObstacleSourceKind.BUSINESS_DECISION:
        return (
            WorkerObstacleKind.BUSINESS_AMBIGUITY,
            ObstacleClassificationReasonCode.BUSINESS_AMBIGUITY,
        )
    if evidence.human_decision_ref is not None:
        return (
            WorkerObstacleKind.HUMAN_DECISION_REQUIRED,
            ObstacleClassificationReasonCode.HUMAN_GATE_PENDING,
        )
    return None, None


def _classify_reliability(
    evidence: WorkerObstacleEvidence,
) -> tuple[WorkerObstacleKind | None, ObstacleClassificationReasonCode | None]:
    if evidence.rate_limited:
        return (
            WorkerObstacleKind.RATE_LIMITED,
            ObstacleClassificationReasonCode.RATE_LIMITED,
        )
    if (
        evidence.dependency_unavailable
        or evidence.source_kind is WorkerObstacleSourceKind.DEPENDENCY_STATUS
    ):
        return (
            WorkerObstacleKind.DEPENDENCY_UNAVAILABLE,
            ObstacleClassificationReasonCode.DEPENDENCY_UNAVAILABLE,
        )
    if evidence.failure_class is FailureClass.POLICY_ERROR:
        return WorkerObstacleKind.POLICY_DENIED, ObstacleClassificationReasonCode.POLICY_DENIED
    if evidence.failure_class is FailureClass.USER_ERROR:
        return (
            WorkerObstacleKind.BUSINESS_AMBIGUITY,
            ObstacleClassificationReasonCode.BUSINESS_AMBIGUITY,
        )
    if evidence.failure_class is FailureClass.QUALITY_ERROR:
        return (
            WorkerObstacleKind.ALTERNATIVE_PATH_AVAILABLE,
            ObstacleClassificationReasonCode.ALTERNATIVE_PATH_AVAILABLE,
        )
    if evidence.runtime_error_code in _TRANSIENT_RUNTIME_ERROR_CODES:
        return (
            WorkerObstacleKind.TRANSIENT_FAILURE,
            ObstacleClassificationReasonCode.TRANSIENT_RUNTIME_FAILURE,
        )
    if evidence.failure_class is FailureClass.RUNTIME_ERROR:
        return (
            WorkerObstacleKind.TRANSIENT_FAILURE,
            ObstacleClassificationReasonCode.TRANSIENT_RUNTIME_FAILURE,
        )
    if evidence.failure_class is FailureClass.DEPENDENCY_ERROR:
        if evidence.retry_after is not None:
            return (
                WorkerObstacleKind.RATE_LIMITED,
                ObstacleClassificationReasonCode.RATE_LIMITED,
            )
        return (
            WorkerObstacleKind.DEPENDENCY_UNAVAILABLE,
            ObstacleClassificationReasonCode.DEPENDENCY_UNAVAILABLE,
        )
    if evidence.runtime_error_code == "dependency_error":
        return (
            WorkerObstacleKind.DEPENDENCY_UNAVAILABLE,
            ObstacleClassificationReasonCode.DEPENDENCY_UNAVAILABLE,
        )
    return None, None
