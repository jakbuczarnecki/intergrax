# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bounded worker recovery decision service (AW-6A).

Accepts obstacle evidence, classifies deterministically, resolves allowed
strategy, enforces safety invariants, and returns a typed decision.

Does not execute recovery, grant authority, schedule work, or invoke LLM.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final, Sequence

from intergrax.autonomous_work.obstacle_recovery_ports import CapabilityAcquisitionPolicy
from intergrax.autonomous_work.worker_obstacle_classifier import (
    CanonicalWorkerObstacleClassifier,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    DECISION_POLICY_VERSION,
    ObstacleClassificationDisposition,
    ObstacleClassificationReasonCode,
    RecoveryDecisionReasonCode,
    RecoveryStrategy,
    WorkerObstacleClassifier,
    WorkerObstacleClassification,
    WorkerObstacleEvidence,
    WorkerObstacleKind,
    WorkerRecoveryDecision,
    WorkerRecoveryDecisionContext,
    WorkerRecoveryDecisionResult,
    derive_recovery_decision_id,
    derive_worker_obstacle_id,
    is_safety_critical_obstacle_kind,
)
from intergrax.contracts.resilience_policy import ResiliencePolicy

_DEFAULT_MAX_RETRY_ATTEMPTS: Final[int] = 3


class WorkerRecoveryDecisionService:
    """Evidence → classification → bounded recovery decision."""

    def __init__(
        self,
        *,
        canonical_classifier: CanonicalWorkerObstacleClassifier | None = None,
        domain_classifiers: Sequence[WorkerObstacleClassifier] = (),
        capability_policy: CapabilityAcquisitionPolicy | None = None,
        resilience_policy: ResiliencePolicy | None = None,
    ) -> None:
        self._canonical_classifier = canonical_classifier or CanonicalWorkerObstacleClassifier()
        self._domain_classifiers = tuple(domain_classifiers)
        self._capability_policy = capability_policy
        self._resilience_policy = resilience_policy

    def decide(
        self,
        evidence: WorkerObstacleEvidence,
        *,
        context: WorkerRecoveryDecisionContext | None = None,
        decided_at: datetime | None = None,
    ) -> WorkerRecoveryDecisionResult:
        resolved_context = context or WorkerRecoveryDecisionContext()
        timestamp = decided_at or evidence.observed_at
        classification_result = self._classify_with_chain(evidence, classified_at=timestamp)
        if classification_result.disposition is not ObstacleClassificationDisposition.CLASSIFIED:
            return classification_result

        classification = classification_result.classification
        assert classification is not None
        decision = self._resolve_decision(
            evidence=evidence,
            classification=classification,
            context=resolved_context,
            decided_at=timestamp,
        )
        return WorkerRecoveryDecisionResult(
            disposition=ObstacleClassificationDisposition.CLASSIFIED,
            classification=classification,
            decision=decision,
        )

    def _classify_with_chain(
        self,
        evidence: WorkerObstacleEvidence,
        *,
        classified_at: datetime,
    ) -> WorkerRecoveryDecisionResult:
        canonical = self._canonical_classifier.classify(
            evidence,
            classified_at=classified_at,
        )
        domain_classification = None
        for classifier in self._domain_classifiers:
            candidate = classifier.classify(evidence, classified_at=classified_at)
            if candidate is None:
                continue
            if domain_classification is None:
                domain_classification = candidate
                continue
            if candidate.obstacle_kind is not domain_classification.obstacle_kind:
                return _conflict_result(
                    evidence=evidence,
                    classified_at=classified_at,
                    reason_code=ObstacleClassificationReasonCode.CLASSIFIER_CONFLICT,
                )
        if domain_classification is not None:
            if domain_classification.obstacle_kind is not canonical.obstacle_kind:
                if (
                    is_safety_critical_obstacle_kind(canonical.obstacle_kind)
                    or is_safety_critical_obstacle_kind(domain_classification.obstacle_kind)
                ):
                    return _conflict_result(
                        evidence=evidence,
                        classified_at=classified_at,
                        reason_code=ObstacleClassificationReasonCode.CLASSIFIER_CONFLICT,
                    )
        return WorkerRecoveryDecisionResult(
            disposition=ObstacleClassificationDisposition.CLASSIFIED,
            classification=canonical,
            decision=None,
        )

    def _resolve_decision(
        self,
        *,
        evidence: WorkerObstacleEvidence,
        classification,
        context: WorkerRecoveryDecisionContext,
        decided_at: datetime,
    ) -> WorkerRecoveryDecision:
        obstacle_id = classification.obstacle_id
        decision_id = derive_recovery_decision_id(
            obstacle_id,
            decision_policy_version=context.decision_policy_version,
        )
        resume_target_ref = _resume_target_ref(evidence)
        evidence_refs = classification.evidence_refs

        if classification.obstacle_kind is WorkerObstacleKind.POLICY_DENIED:
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.STOP,
                decision_reason_code=RecoveryDecisionReasonCode.POLICY_DENY_STOP,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
            )

        if classification.obstacle_kind is WorkerObstacleKind.CREDENTIAL_UNAVAILABLE:
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.ESCALATE,
                decision_reason_code=RecoveryDecisionReasonCode.CREDENTIAL_ESCALATE,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
            )

        if classification.obstacle_kind is WorkerObstacleKind.HUMAN_DECISION_REQUIRED:
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.REQUEST_HUMAN_DECISION,
                decision_reason_code=RecoveryDecisionReasonCode.HUMAN_DECISION_REQUIRED,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
                human_decision_ref=evidence.human_decision_ref,
                recommended_worker_state=WorkerLifecycleState.WAITING_FOR_HUMAN,
            )

        if classification.obstacle_kind is WorkerObstacleKind.BUSINESS_AMBIGUITY:
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.REQUEST_HUMAN_DECISION,
                decision_reason_code=RecoveryDecisionReasonCode.BUSINESS_AMBIGUITY_HUMAN,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
                human_decision_ref=evidence.human_decision_ref,
                recommended_worker_state=WorkerLifecycleState.WAITING_FOR_HUMAN,
            )

        if classification.obstacle_kind is WorkerObstacleKind.TRANSIENT_FAILURE:
            max_attempts = _resolve_max_attempts(context, self._resilience_policy)
            if max_attempts is None or max_attempts <= 0:
                return WorkerRecoveryDecision(
                    decision_id=decision_id,
                    obstacle_id=obstacle_id,
                    obstacle_kind=classification.obstacle_kind,
                    strategy=RecoveryStrategy.ESCALATE,
                    decision_reason_code=(
                        RecoveryDecisionReasonCode.TRANSIENT_RETRY_UNBOUNDED_ESCALATE
                    ),
                    evidence_refs=evidence_refs,
                    decided_at=decided_at,
                    source_ref=evidence.source_ref,
                    decision_policy_version=context.decision_policy_version,
                    resume_target_ref=resume_target_ref,
                )
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.RETRY,
                decision_reason_code=RecoveryDecisionReasonCode.TRANSIENT_RETRY_BOUNDED,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
                retry_after=evidence.retry_after,
                max_attempts=max_attempts,
                recommended_worker_state=WorkerLifecycleState.RECOVERING,
            )

        if classification.obstacle_kind is WorkerObstacleKind.DEPENDENCY_UNAVAILABLE:
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.WAIT,
                decision_reason_code=RecoveryDecisionReasonCode.DEPENDENCY_WAIT,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
                retry_after=evidence.retry_after,
                dependency_ref=evidence.dependency_ref,
                recommended_worker_state=WorkerLifecycleState.WAITING_EXTERNAL,
            )

        if classification.obstacle_kind is WorkerObstacleKind.RATE_LIMITED:
            strategy = (
                RecoveryStrategy.WAIT
                if evidence.retry_after is not None
                else RecoveryStrategy.THROTTLE
            )
            reason = (
                RecoveryDecisionReasonCode.RATE_LIMIT_WAIT
                if strategy is RecoveryStrategy.WAIT
                else RecoveryDecisionReasonCode.RATE_LIMIT_THROTTLE
            )
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=strategy,
                decision_reason_code=reason,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
                retry_after=evidence.retry_after,
                recommended_worker_state=WorkerLifecycleState.WAITING_EXTERNAL,
            )

        if classification.obstacle_kind is WorkerObstacleKind.ALTERNATIVE_PATH_AVAILABLE:
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.REPLAN,
                decision_reason_code=RecoveryDecisionReasonCode.ALTERNATIVE_PATH_REPLAN,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
                recommended_worker_state=WorkerLifecycleState.RECOVERING,
            )

        if classification.obstacle_kind is WorkerObstacleKind.SCHEMA_OR_API_DRIFT:
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.ADAPT_INTEGRATION,
                decision_reason_code=RecoveryDecisionReasonCode.SCHEMA_DRIFT_ADAPT,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
                recommended_worker_state=WorkerLifecycleState.RECOVERING,
            )

        if classification.obstacle_kind is WorkerObstacleKind.CAPABILITY_MISSING:
            allowed = _capability_acquisition_allowed(
                evidence=evidence,
                context=context,
                capability_policy=self._capability_policy,
            )
            if allowed:
                return WorkerRecoveryDecision(
                    decision_id=decision_id,
                    obstacle_id=obstacle_id,
                    obstacle_kind=classification.obstacle_kind,
                    strategy=RecoveryStrategy.ACQUIRE_CAPABILITY,
                    decision_reason_code=RecoveryDecisionReasonCode.CAPABILITY_ACQUIRE_ALLOWED,
                    evidence_refs=evidence_refs,
                    decided_at=decided_at,
                    source_ref=evidence.source_ref,
                    decision_policy_version=context.decision_policy_version,
                    resume_target_ref=resume_target_ref,
                    recommended_worker_state=WorkerLifecycleState.RECOVERING,
                )
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.ESCALATE,
                decision_reason_code=RecoveryDecisionReasonCode.CAPABILITY_ACQUIRE_DENIED_ESCALATE,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
            )

        if classification.obstacle_kind is WorkerObstacleKind.SUSPICIOUS_OR_UNSAFE:
            return WorkerRecoveryDecision(
                decision_id=decision_id,
                obstacle_id=obstacle_id,
                obstacle_kind=classification.obstacle_kind,
                strategy=RecoveryStrategy.QUARANTINE,
                decision_reason_code=RecoveryDecisionReasonCode.SUSPICIOUS_QUARANTINE,
                evidence_refs=evidence_refs,
                decided_at=decided_at,
                source_ref=evidence.source_ref,
                decision_policy_version=context.decision_policy_version,
                resume_target_ref=resume_target_ref,
                recommended_worker_state=WorkerLifecycleState.QUARANTINED,
            )

        return WorkerRecoveryDecision(
            decision_id=decision_id,
            obstacle_id=obstacle_id,
            obstacle_kind=classification.obstacle_kind,
            strategy=RecoveryStrategy.ESCALATE,
            decision_reason_code=RecoveryDecisionReasonCode.UNKNOWN_ESCALATE,
            evidence_refs=evidence_refs,
            decided_at=decided_at,
            source_ref=evidence.source_ref,
            decision_policy_version=context.decision_policy_version,
            resume_target_ref=resume_target_ref,
        )


def _resolve_max_attempts(
    context: WorkerRecoveryDecisionContext,
    resilience_policy: ResiliencePolicy | None,
) -> int | None:
    if context.max_retry_attempts is not None:
        return context.max_retry_attempts
    if resilience_policy is not None:
        return resilience_policy.max_attempts
    return _DEFAULT_MAX_RETRY_ATTEMPTS


def _capability_acquisition_allowed(
    *,
    evidence: WorkerObstacleEvidence,
    context: WorkerRecoveryDecisionContext,
    capability_policy: CapabilityAcquisitionPolicy | None,
) -> bool:
    if capability_policy is not None and evidence.capability_profile_ref is not None:
        return capability_policy.is_acquisition_allowed(evidence.capability_profile_ref)
    return context.capability_acquisition_allowed


def _resume_target_ref(evidence: WorkerObstacleEvidence) -> str | None:
    if evidence.execution_id is not None:
        return str(evidence.execution_id)
    if evidence.run_id is not None:
        return str(evidence.run_id)
    if evidence.goal_id is not None:
        return str(evidence.goal_id)
    return None


def _conflict_result(
    *,
    evidence: WorkerObstacleEvidence,
    classified_at: datetime,
    reason_code: ObstacleClassificationReasonCode,
) -> WorkerRecoveryDecisionResult:
    obstacle_id = derive_worker_obstacle_id(evidence)
    decision_id = derive_recovery_decision_id(obstacle_id)
    classification = WorkerObstacleClassification(
        obstacle_id=obstacle_id,
        obstacle_kind=WorkerObstacleKind.UNKNOWN,
        classifier_id="canonical.deterministic.v1",
        reason_code=reason_code,
        evidence_refs=evidence.problem_evidence_refs,
        classified_at=classified_at,
    )
    decision = WorkerRecoveryDecision(
        decision_id=decision_id,
        obstacle_id=obstacle_id,
        obstacle_kind=WorkerObstacleKind.UNKNOWN,
        strategy=RecoveryStrategy.ESCALATE,
        decision_reason_code=RecoveryDecisionReasonCode.CLASSIFICATION_CONFLICT_ESCALATE,
        evidence_refs=evidence.problem_evidence_refs,
        decided_at=classified_at,
        source_ref=evidence.source_ref,
        decision_policy_version=DECISION_POLICY_VERSION,
    )
    return WorkerRecoveryDecisionResult(
        disposition=ObstacleClassificationDisposition.CONFLICT,
        classification=classification,
        decision=decision,
    )
