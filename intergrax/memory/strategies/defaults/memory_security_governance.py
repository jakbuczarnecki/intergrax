# © Artur Czarnecki. All rights reserved.

"""Default memory security & governance strategies (MEM-ENT-10)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryRecordSourceType,
    MemoryTrustClass,
)
from intergrax.memory.contracts.memory_control import MemoryControlPlaneScope
from intergrax.memory.contracts.memory_security_governance import (
    MemoryAdmissionPolicy,
    MemoryAuthorizationPolicy,
    MemoryGovernanceDecision,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernancePolicy,
    MemoryGovernanceReasonCode,
    MemoryRetentionAction,
    MemoryRetentionDecision,
    MemoryRetentionPolicy,
    MemorySecurityStrategySet,
    MemoryTrustEvaluationPolicy,
    MemoryTrustEvaluationResult,
)

_POLICY_VERSION = "1.0.0"

_TRUST_RANK: dict[MemoryTrustClass, int] = {
    MemoryTrustClass.UNKNOWN: 0,
    MemoryTrustClass.MODEL_INFERENCE: 1,
    MemoryTrustClass.SYSTEM_GENERATED: 2,
    MemoryTrustClass.EXTERNAL_SOURCE: 3,
    MemoryTrustClass.USER_EXPLICIT: 4,
}

_MAX_TRUST_BY_SOURCE: dict[MemoryRecordSourceType, MemoryTrustClass] = {
    MemoryRecordSourceType.USER_EXPLICIT: MemoryTrustClass.USER_EXPLICIT,
    MemoryRecordSourceType.SESSION_EXTRACTION: MemoryTrustClass.MODEL_INFERENCE,
    MemoryRecordSourceType.TOOL_RESULT: MemoryTrustClass.EXTERNAL_SOURCE,
    MemoryRecordSourceType.SYSTEM: MemoryTrustClass.SYSTEM_GENERATED,
    MemoryRecordSourceType.IMPORT: MemoryTrustClass.EXTERNAL_SOURCE,
    MemoryRecordSourceType.UNKNOWN: MemoryTrustClass.UNKNOWN,
}


def _allow_decision(
    *,
    policy_id: str,
    operation: MemoryGovernanceOperation,
    subject_memory_id: str | None = None,
    trust_class: MemoryTrustClass | None = None,
    data_classification: DataClassification | None = None,
) -> MemoryGovernanceDecision:
    return MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.ALLOW,
        reason_code=MemoryGovernanceReasonCode.ALLOWED,
        policy_id=policy_id,
        policy_version=_POLICY_VERSION,
        operation=operation,
        trust_class=trust_class,
        data_classification=data_classification,
        subject_memory_id=subject_memory_id,
    )


def _deny_decision(
    *,
    policy_id: str,
    operation: MemoryGovernanceOperation,
    reason_code: MemoryGovernanceReasonCode,
    subject_memory_id: str | None = None,
) -> MemoryGovernanceDecision:
    return MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=reason_code,
        policy_id=policy_id,
        policy_version=_POLICY_VERSION,
        operation=operation,
        subject_memory_id=subject_memory_id,
    )


@dataclass(frozen=True, slots=True)
class DefaultMemoryAuthorizationPolicy:
    policy_id: str = "memory.authorization.default"
    policy_version: str = _POLICY_VERSION

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        scope = request.context.scope
        identity = request.context.identity
        if scope.tenant_id != identity.tenant_id:
            return _deny_decision(
                policy_id=self.policy_id,
                operation=request.context.operation,
                reason_code=MemoryGovernanceReasonCode.CROSS_SCOPE,
            )
        if scope.kind is MemoryControlPlaneScope.USER:
            canonical_user = (identity.user_id or "").strip()
            scope_user = (scope.user_id or "").strip()
            if not scope_user or scope_user != canonical_user:
                return _deny_decision(
                    policy_id=self.policy_id,
                    operation=request.context.operation,
                    reason_code=MemoryGovernanceReasonCode.CROSS_SCOPE,
                )
        return _allow_decision(
            policy_id=self.policy_id,
            operation=request.context.operation,
            subject_memory_id=(
                request.proposed_record.memory_id
                if request.proposed_record is not None
                else request.target.memory_id if request.target is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class DefaultMemoryTrustEvaluationPolicy:
    policy_id: str = "memory.trust.default"
    policy_version: str = _POLICY_VERSION

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryTrustEvaluationResult:
        record = request.proposed_record or (
            request.existing_record
            if request.existing_record is not None
            else None
        )
        if record is None:
            return MemoryTrustEvaluationResult(
                effective_trust_class=MemoryTrustClass.UNKNOWN,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
            )
        declared = record.trust.trust_class
        max_allowed = _MAX_TRUST_BY_SOURCE.get(
            record.provenance.source_type,
            MemoryTrustClass.UNKNOWN,
        )
        if _TRUST_RANK[declared] > _TRUST_RANK[max_allowed]:
            capped = max_allowed
            return MemoryTrustEvaluationResult(
                effective_trust_class=capped,
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                escalation_blocked=True,
                reason_code=MemoryGovernanceReasonCode.TRUST_ESCALATION_BLOCKED,
            )
        for source in request.source_records:
            source_trust = source.trust.trust_class
            if _TRUST_RANK[declared] > _TRUST_RANK[source_trust]:
                return MemoryTrustEvaluationResult(
                    effective_trust_class=source_trust,
                    policy_id=self.policy_id,
                    policy_version=self.policy_version,
                    escalation_blocked=True,
                    reason_code=MemoryGovernanceReasonCode.TRUST_ESCALATION_BLOCKED,
                )
        return MemoryTrustEvaluationResult(
            effective_trust_class=declared,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )


@dataclass(frozen=True, slots=True)
class DefaultMemoryAdmissionPolicy:
    policy_id: str = "memory.admission.default"
    policy_version: str = _POLICY_VERSION

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        operation = request.context.operation
        subject_id = None
        record = request.proposed_record
        if record is not None:
            subject_id = record.memory_id
            if (
                record.provenance.source_type is MemoryRecordSourceType.UNKNOWN
                and record.trust.trust_class is MemoryTrustClass.USER_EXPLICIT
            ):
                return _deny_decision(
                    policy_id=self.policy_id,
                    operation=operation,
                    reason_code=MemoryGovernanceReasonCode.POISONING_SUSPECTED,
                    subject_memory_id=subject_id,
                )
            if record.trust.trust_class is MemoryTrustClass.MODEL_INFERENCE:
                if operation in {
                    MemoryGovernanceOperation.REMEMBER,
                    MemoryGovernanceOperation.PROMOTE,
                    MemoryGovernanceOperation.PROJECT,
                }:
                    if record.provenance.source_type not in {
                        MemoryRecordSourceType.SESSION_EXTRACTION,
                        MemoryRecordSourceType.SYSTEM,
                        MemoryRecordSourceType.TOOL_RESULT,
                    }:
                        return MemoryGovernanceDecision(
                            outcome=MemoryGovernanceOutcome.REQUIRE_REVIEW,
                            reason_code=MemoryGovernanceReasonCode.REVIEW_REQUIRED,
                            policy_id=self.policy_id,
                            policy_version=self.policy_version,
                            operation=operation,
                            subject_memory_id=subject_id,
                        )
        return _allow_decision(
            policy_id=self.policy_id,
            operation=operation,
            subject_memory_id=subject_id,
        )


@dataclass(frozen=True, slots=True)
class DefaultMemoryGovernancePolicy:
    policy_id: str = "memory.governance.default"
    policy_version: str = _POLICY_VERSION

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        operation = request.context.operation
        record = request.proposed_record or request.existing_record
        if record is None:
            return _allow_decision(policy_id=self.policy_id, operation=operation)
        classification = record.governance.data_classification
        subject_id = record.memory_id
        if classification is DataClassification.RESTRICTED:
            if operation in {
                MemoryGovernanceOperation.REMEMBER,
                MemoryGovernanceOperation.PROMOTE,
                MemoryGovernanceOperation.PROJECT,
                MemoryGovernanceOperation.COMPACT,
            }:
                return _deny_decision(
                    policy_id=self.policy_id,
                    operation=operation,
                    reason_code=MemoryGovernanceReasonCode.SENSITIVE_DATA_POLICY,
                    subject_memory_id=subject_id,
                )
            if operation is MemoryGovernanceOperation.RECALL:
                return _deny_decision(
                    policy_id=self.policy_id,
                    operation=operation,
                    reason_code=MemoryGovernanceReasonCode.SENSITIVE_DATA_POLICY,
                    subject_memory_id=subject_id,
                )
        return _allow_decision(
            policy_id=self.policy_id,
            operation=operation,
            subject_memory_id=subject_id,
            data_classification=classification,
            trust_class=record.trust.trust_class,
        )


@dataclass(frozen=True, slots=True)
class DefaultMemoryRetentionPolicy:
    policy_id: str = "memory.retention.default"
    policy_version: str = _POLICY_VERSION

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryRetentionDecision:
        return MemoryRetentionDecision(
            action=MemoryRetentionAction.KEEP,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )


def build_default_memory_security_strategy_set() -> MemorySecurityStrategySet:
    return MemorySecurityStrategySet(
        authorization=DefaultMemoryAuthorizationPolicy(),
        trust=DefaultMemoryTrustEvaluationPolicy(),
        admission=DefaultMemoryAdmissionPolicy(),
        governance=DefaultMemoryGovernancePolicy(),
        retention=DefaultMemoryRetentionPolicy(),
    )
