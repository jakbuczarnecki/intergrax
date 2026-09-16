# © Artur Czarnecki. All rights reserved.

"""Memory security & governance contracts (MEM-ENT-10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.data_classification import DataClassification
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.memory_control import MemoryControlScopeRef
from intergrax.memory.contracts.memory_models import MemoryKind, UserProfileMemoryEntry

__all__ = [
    "MemoryAdmissionPolicy",
    "MemoryAuthorizationPolicy",
    "MemoryGovernanceConstraint",
    "MemoryGovernanceDecision",
    "MemoryGovernanceEvaluationRequest",
    "MemoryGovernanceOperation",
    "MemoryGovernanceOutcome",
    "MemoryGovernancePolicy",
    "MemoryGovernanceReasonCode",
    "MemoryGovernanceRecordSnapshot",
    "MemoryGovernanceTarget",
    "MemoryRetentionAction",
    "MemoryRetentionDecision",
    "MemoryRetentionPolicy",
    "MemorySecurityContext",
    "MemorySecurityStrategySet",
    "MemoryTrustEvaluationPolicy",
    "MemoryTrustEvaluationResult",
]


class MemoryGovernanceOperation(str, Enum):
    REMEMBER = "remember"
    RECALL = "recall"
    PROMOTE = "promote"
    SUPERSEDE = "supersede"
    DELETE = "delete"
    COMPACT = "compact"
    PROJECT = "project"
    UPDATE = "update"


class MemoryGovernanceOutcome(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_WITH_CONSTRAINTS = "allow_with_constraints"
    REQUIRE_REVIEW = "require_review"


class MemoryGovernanceReasonCode(str, Enum):
    ALLOWED = "allowed"
    CROSS_SCOPE = "cross_scope"
    INSUFFICIENT_TRUST = "insufficient_trust"
    TRUST_ESCALATION_BLOCKED = "trust_escalation_blocked"
    POISONING_SUSPECTED = "poisoning_suspected"
    SENSITIVE_DATA_POLICY = "sensitive_data_policy"
    RETENTION_BLOCK = "retention_block"
    GOVERNANCE_DENY = "governance_deny"
    REVIEW_REQUIRED = "review_required"
    AUTHORIZATION_DENY = "authorization_deny"
    POLICY_FAILURE = "policy_failure"
    POLICY_MISSING = "policy_missing"


class MemoryRetentionAction(str, Enum):
    KEEP = "keep"
    EXPIRE_AT = "expire_at"
    DELETE = "delete"
    ARCHIVE = "archive"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class MemoryGovernanceConstraint:
    code: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryGovernanceDecision:
    outcome: MemoryGovernanceOutcome
    reason_code: MemoryGovernanceReasonCode
    policy_id: str
    policy_version: str
    operation: MemoryGovernanceOperation
    trust_class: MemoryTrustClass | None = None
    data_classification: DataClassification | None = None
    retention_action: MemoryRetentionAction | None = None
    constraints: tuple[MemoryGovernanceConstraint, ...] = ()
    subject_memory_id: str | None = None

    def permits_mutation(self) -> bool:
        return self.outcome in {
            MemoryGovernanceOutcome.ALLOW,
            MemoryGovernanceOutcome.ALLOW_WITH_CONSTRAINTS,
        }

    def permits_disclosure(self) -> bool:
        return self.outcome in {
            MemoryGovernanceOutcome.ALLOW,
            MemoryGovernanceOutcome.ALLOW_WITH_CONSTRAINTS,
        }


@dataclass(frozen=True, slots=True)
class MemoryGovernanceTarget:
    memory_id: str
    revision: int | None = None
    kind: MemoryKind | None = None
    scope: MemoryControlScopeRef | None = None


@dataclass(frozen=True, slots=True)
class MemoryGovernanceRecordSnapshot:
    memory_id: str
    revision: int
    kind: MemoryKind
    provenance: MemoryProvenance
    trust: MemoryRecordTrust
    governance: MemoryRecordGovernance
    content_preview: str | None = None

    @classmethod
    def from_user_profile_entry(cls, entry: UserProfileMemoryEntry) -> MemoryGovernanceRecordSnapshot:
        preview = entry.content[:256] if entry.content else None
        return cls(
            memory_id=entry.entry_id,
            revision=entry.revision,
            kind=entry.kind,
            provenance=entry.provenance,
            trust=entry.trust,
            governance=entry.governance,
            content_preview=preview,
        )


@dataclass(frozen=True, slots=True)
class MemorySecurityContext:
    identity: RequestIdentity
    scope: MemoryControlScopeRef
    operation: MemoryGovernanceOperation
    reference_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemoryGovernanceEvaluationRequest:
    context: MemorySecurityContext
    target: MemoryGovernanceTarget | None = None
    proposed_record: MemoryGovernanceRecordSnapshot | None = None
    existing_record: MemoryGovernanceRecordSnapshot | None = None
    source_records: tuple[MemoryGovernanceRecordSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryRetentionDecision:
    action: MemoryRetentionAction
    policy_id: str
    policy_version: str
    expire_at_iso: str | None = None
    reason_code: MemoryGovernanceReasonCode = MemoryGovernanceReasonCode.ALLOWED


@dataclass(frozen=True, slots=True)
class MemoryTrustEvaluationResult:
    effective_trust_class: MemoryTrustClass
    policy_id: str
    policy_version: str
    escalation_blocked: bool = False
    reason_code: MemoryGovernanceReasonCode = MemoryGovernanceReasonCode.ALLOWED


@dataclass(frozen=True, slots=True)
class MemorySecurityStrategySet:
    authorization: MemoryAuthorizationPolicy
    trust: MemoryTrustEvaluationPolicy
    admission: MemoryAdmissionPolicy
    governance: MemoryGovernancePolicy
    retention: MemoryRetentionPolicy


class MemoryAuthorizationPolicy(Protocol):
    policy_id: str
    policy_version: str

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        """Authorize the operation for actor and scope."""


class MemoryTrustEvaluationPolicy(Protocol):
    policy_id: str
    policy_version: str

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryTrustEvaluationResult:
        """Evaluate effective trust; block silent escalation."""


class MemoryAdmissionPolicy(Protocol):
    policy_id: str
    policy_version: str

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        """Admission / poisoning boundary for writes and promotions."""


class MemoryGovernancePolicy(Protocol):
    policy_id: str
    policy_version: str

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        """Sensitivity and governance metadata enforcement."""


class MemoryRetentionPolicy(Protocol):
    policy_id: str
    policy_version: str

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryRetentionDecision:
        """Retention posture (decision only; no scheduler)."""
