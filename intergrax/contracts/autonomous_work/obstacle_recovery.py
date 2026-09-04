# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical worker obstacle taxonomy and recovery decision contracts (AW-6A).

Produces obstacle evidence → classification → bounded recovery decision.
Does not execute recovery, mint authority, or invoke LLM classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerGoalId,
    WorkerInstanceId,
    validate_responsibility_id,
    validate_worker_goal_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.lifecycle import WorkerLifecycleState
from intergrax.contracts.autonomous_work.profile_reference import CapabilityProfileRef
from intergrax.contracts.autonomous_work.references import (
    ExternalDependencyReference,
    HumanPendingReference,
    ProblemReference,
    validate_external_dependency_reference,
    validate_human_pending_reference,
    validate_problem_reference,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)
from intergrax.contracts.policy_action import PolicyAction
from intergrax.contracts.resilience_policy import FailureClass
from intergrax.contracts.runtime_policy import PolicyDecision

DECISION_POLICY_VERSION: str = "aw-6a.v1"

WORKER_OBSTACLE_TAXONOMY: tuple[str, ...] = (
    "TRANSIENT_FAILURE",
    "DEPENDENCY_UNAVAILABLE",
    "RATE_LIMITED",
    "CREDENTIAL_UNAVAILABLE",
    "POLICY_DENIED",
    "HUMAN_DECISION_REQUIRED",
    "BUSINESS_AMBIGUITY",
    "ALTERNATIVE_PATH_AVAILABLE",
    "SCHEMA_OR_API_DRIFT",
    "CAPABILITY_MISSING",
    "SUSPICIOUS_OR_UNSAFE",
    "UNKNOWN",
)

_RUNTIME_ERROR_SIGNALS: frozenset[str] = frozenset(
    {
        "internal_error",
        "validation_error",
        "timeout",
        "llm_error",
        "tool_error",
        "user_error",
        "policy_error",
        "dependency_error",
        "quality_error",
        "permission_error",
        "runtime_error",
    }
)


def validate_runtime_error_signal(value: object) -> str:
    signal = require_non_empty_text(value, label="runtime_error_code")
    if signal not in _RUNTIME_ERROR_SIGNALS:
        raise ValueError(f"runtime_error_code must be a known runtime signal, got {signal!r}")
    return signal

CANONICAL_RECOVERY_STRATEGIES: tuple[str, ...] = (
    "RETRY",
    "WAIT",
    "THROTTLE",
    "REPLAN",
    "REQUEST_HUMAN_DECISION",
    "ESCALATE",
    "ACQUIRE_CAPABILITY",
    "ADAPT_INTEGRATION",
    "QUARANTINE",
    "STOP",
)

class WorkerObstacleSourceKind(StrEnum):
    """Implemented obstacle evidence sources only."""

    EXECUTION_FAILURE = "execution_failure"
    POLICY_DECISION = "policy_decision"
    DEPENDENCY_STATUS = "dependency_status"
    HUMAN_GATE = "human_gate"
    BUSINESS_DECISION = "business_decision"
    CAPABILITY_RESOLUTION = "capability_resolution"
    OPERATOR = "operator"


class WorkerObstacleKind(StrEnum):
    """Canonical worker-level obstacle taxonomy — not exceptions."""

    TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    CREDENTIAL_UNAVAILABLE = "CREDENTIAL_UNAVAILABLE"
    POLICY_DENIED = "POLICY_DENIED"
    HUMAN_DECISION_REQUIRED = "HUMAN_DECISION_REQUIRED"
    BUSINESS_AMBIGUITY = "BUSINESS_AMBIGUITY"
    ALTERNATIVE_PATH_AVAILABLE = "ALTERNATIVE_PATH_AVAILABLE"
    SCHEMA_OR_API_DRIFT = "SCHEMA_OR_API_DRIFT"
    CAPABILITY_MISSING = "CAPABILITY_MISSING"
    SUSPICIOUS_OR_UNSAFE = "SUSPICIOUS_OR_UNSAFE"
    UNKNOWN = "UNKNOWN"


class RecoveryStrategy(StrEnum):
    """Bounded recovery strategy recommendation — execution belongs to AW-6B."""

    RETRY = "RETRY"
    WAIT = "WAIT"
    THROTTLE = "THROTTLE"
    REPLAN = "REPLAN"
    REQUEST_HUMAN_DECISION = "REQUEST_HUMAN_DECISION"
    ESCALATE = "ESCALATE"
    ACQUIRE_CAPABILITY = "ACQUIRE_CAPABILITY"
    ADAPT_INTEGRATION = "ADAPT_INTEGRATION"
    QUARANTINE = "QUARANTINE"
    STOP = "STOP"


class ObstacleClassificationDisposition(StrEnum):
    """Typed classification service outcome."""

    CLASSIFIED = "CLASSIFIED"
    UNCLASSIFIABLE = "UNCLASSIFIABLE"
    CONFLICT = "CONFLICT"
    UNAVAILABLE = "UNAVAILABLE"


class ObstacleClassificationReasonCode(StrEnum):
    """Evidence-bearing obstacle classification reason codes."""

    POLICY_DENIED = "POLICY_DENIED"
    POLICY_REQUIRE_HUMAN = "POLICY_REQUIRE_HUMAN"
    POLICY_ESCALATE = "POLICY_ESCALATE"
    POLICY_MODIFY_UNRESOLVED = "POLICY_MODIFY_UNRESOLVED"
    CREDENTIAL_UNAVAILABLE = "CREDENTIAL_UNAVAILABLE"
    HUMAN_GATE_PENDING = "HUMAN_GATE_PENDING"
    BUSINESS_AMBIGUITY = "BUSINESS_AMBIGUITY"
    TRANSIENT_RUNTIME_FAILURE = "TRANSIENT_RUNTIME_FAILURE"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    ALTERNATIVE_PATH_AVAILABLE = "ALTERNATIVE_PATH_AVAILABLE"
    SCHEMA_OR_API_DRIFT = "SCHEMA_OR_API_DRIFT"
    CAPABILITY_MISSING = "CAPABILITY_MISSING"
    SUSPICIOUS_OR_UNSAFE = "SUSPICIOUS_OR_UNSAFE"
    UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE"
    CLASSIFIER_CONFLICT = "CLASSIFIER_CONFLICT"


class RecoveryDecisionReasonCode(StrEnum):
    """Evidence-bearing recovery decision reason codes."""

    CANONICAL_MAPPING = "CANONICAL_MAPPING"
    POLICY_DENY_STOP = "POLICY_DENY_STOP"
    CREDENTIAL_ESCALATE = "CREDENTIAL_ESCALATE"
    HUMAN_DECISION_REQUIRED = "HUMAN_DECISION_REQUIRED"
    BUSINESS_AMBIGUITY_HUMAN = "BUSINESS_AMBIGUITY_HUMAN"
    TRANSIENT_RETRY_BOUNDED = "TRANSIENT_RETRY_BOUNDED"
    TRANSIENT_RETRY_UNBOUNDED_ESCALATE = "TRANSIENT_RETRY_UNBOUNDED_ESCALATE"
    DEPENDENCY_WAIT = "DEPENDENCY_WAIT"
    RATE_LIMIT_THROTTLE = "RATE_LIMIT_THROTTLE"
    RATE_LIMIT_WAIT = "RATE_LIMIT_WAIT"
    ALTERNATIVE_PATH_REPLAN = "ALTERNATIVE_PATH_REPLAN"
    SCHEMA_DRIFT_ADAPT = "SCHEMA_DRIFT_ADAPT"
    CAPABILITY_ACQUIRE_ALLOWED = "CAPABILITY_ACQUIRE_ALLOWED"
    CAPABILITY_ACQUIRE_DENIED_ESCALATE = "CAPABILITY_ACQUIRE_DENIED_ESCALATE"
    SUSPICIOUS_QUARANTINE = "SUSPICIOUS_QUARANTINE"
    UNKNOWN_ESCALATE = "UNKNOWN_ESCALATE"
    CLASSIFICATION_CONFLICT_ESCALATE = "CLASSIFICATION_CONFLICT_ESCALATE"
    CLASSIFICATION_UNAVAILABLE = "CLASSIFICATION_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class WorkerObstacleEvidence:
    """Immutable worker obstacle evidence — durable fact, not an exception."""

    worker_instance_id: WorkerInstanceId
    source_kind: WorkerObstacleSourceKind
    source_ref: str
    occurrence_identity: str
    observed_at: datetime
    problem_evidence_refs: tuple[ProblemReference, ...] = ()
    execution_id: ExecutionId | None = None
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    goal_id: WorkerGoalId | None = None
    goal_revision: Revision | None = None
    responsibility_id: ResponsibilityId | None = None
    runtime_error_code: str | None = None
    failure_class: FailureClass | None = None
    policy_decision: PolicyDecision | None = None
    dependency_ref: ExternalDependencyReference | None = None
    credential_ref: str | None = None
    human_decision_ref: HumanPendingReference | None = None
    alternative_path_ref: str | None = None
    capability_missing_ref: str | None = None
    retry_after: datetime | None = None
    dependency_unavailable: bool = False
    rate_limited: bool = False
    schema_drift_detected: bool = False
    suspicious_or_unsafe: bool = False
    business_ambiguity: bool = False
    capability_profile_ref: CapabilityProfileRef | None = None

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.source_kind) is not WorkerObstacleSourceKind:
            raise TypeError("source_kind must be WorkerObstacleSourceKind")
        object.__setattr__(
            self,
            "source_ref",
            require_non_empty_text(self.source_ref, label="source_ref"),
        )
        object.__setattr__(
            self,
            "occurrence_identity",
            require_non_empty_text(
                self.occurrence_identity,
                label="occurrence_identity",
            ),
        )
        object.__setattr__(
            self,
            "observed_at",
            require_aware_utc(self.observed_at, label="observed_at"),
        )
        object.__setattr__(
            self,
            "problem_evidence_refs",
            freeze_tuple(self.problem_evidence_refs, label="problem_evidence_refs"),
        )
        for ref in self.problem_evidence_refs:
            validate_problem_reference(ref)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.goal_id is not None:
            validate_worker_goal_id(self.goal_id)
        if self.goal_revision is not None:
            validate_revision(self.goal_revision)
        if self.responsibility_id is not None:
            validate_responsibility_id(self.responsibility_id)
        if self.runtime_error_code is not None:
            object.__setattr__(
                self,
                "runtime_error_code",
                validate_runtime_error_signal(self.runtime_error_code),
            )
        if self.dependency_ref is not None:
            validate_external_dependency_reference(self.dependency_ref)
        if self.credential_ref is not None:
            object.__setattr__(
                self,
                "credential_ref",
                require_non_empty_text(self.credential_ref, label="credential_ref"),
            )
        if self.human_decision_ref is not None:
            validate_human_pending_reference(self.human_decision_ref)
        if self.alternative_path_ref is not None:
            object.__setattr__(
                self,
                "alternative_path_ref",
                require_non_empty_text(
                    self.alternative_path_ref,
                    label="alternative_path_ref",
                ),
            )
        if self.capability_missing_ref is not None:
            object.__setattr__(
                self,
                "capability_missing_ref",
                require_non_empty_text(
                    self.capability_missing_ref,
                    label="capability_missing_ref",
                ),
            )
        if self.retry_after is not None:
            object.__setattr__(
                self,
                "retry_after",
                require_aware_utc(self.retry_after, label="retry_after"),
            )


def derive_worker_obstacle_id(evidence: WorkerObstacleEvidence) -> str:
    """Stable logical obstacle identity — not a random per-retry UUID."""

    return (
        f"{evidence.worker_instance_id}:"
        f"{evidence.source_kind.value}:"
        f"{evidence.source_ref}:"
        f"{evidence.occurrence_identity}"
    )


def derive_recovery_decision_id(
    obstacle_id: str,
    *,
    decision_policy_version: str = DECISION_POLICY_VERSION,
) -> str:
    return f"{obstacle_id}:{decision_policy_version}"


@dataclass(frozen=True, slots=True)
class WorkerObstacleClassification:
    """Typed obstacle classification result."""

    obstacle_id: str
    obstacle_kind: WorkerObstacleKind
    classifier_id: str
    reason_code: ObstacleClassificationReasonCode
    evidence_refs: tuple[ProblemReference, ...]
    classified_at: datetime
    confidence_category: str = "deterministic"
    explanation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "obstacle_id",
            require_non_empty_text(self.obstacle_id, label="obstacle_id"),
        )
        if type(self.obstacle_kind) is not WorkerObstacleKind:
            raise TypeError("obstacle_kind must be WorkerObstacleKind")
        object.__setattr__(
            self,
            "classifier_id",
            require_non_empty_text(self.classifier_id, label="classifier_id"),
        )
        if type(self.reason_code) is not ObstacleClassificationReasonCode:
            raise TypeError("reason_code must be ObstacleClassificationReasonCode")
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)
        object.__setattr__(
            self,
            "classified_at",
            require_aware_utc(self.classified_at, label="classified_at"),
        )


@dataclass(frozen=True, slots=True)
class WorkerRecoveryDecision:
    """Immutable bounded recovery decision — recommendation only."""

    decision_id: str
    obstacle_id: str
    obstacle_kind: WorkerObstacleKind
    strategy: RecoveryStrategy
    decision_reason_code: RecoveryDecisionReasonCode
    evidence_refs: tuple[ProblemReference, ...]
    decided_at: datetime
    source_ref: str
    decision_policy_version: str = DECISION_POLICY_VERSION
    retry_after: datetime | None = None
    max_attempts: int | None = None
    resume_target_ref: str | None = None
    human_decision_ref: HumanPendingReference | None = None
    dependency_ref: ExternalDependencyReference | None = None
    recommended_worker_state: WorkerLifecycleState | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_id",
            require_non_empty_text(self.decision_id, label="decision_id"),
        )
        object.__setattr__(
            self,
            "obstacle_id",
            require_non_empty_text(self.obstacle_id, label="obstacle_id"),
        )
        if type(self.obstacle_kind) is not WorkerObstacleKind:
            raise TypeError("obstacle_kind must be WorkerObstacleKind")
        if type(self.strategy) is not RecoveryStrategy:
            raise TypeError("strategy must be RecoveryStrategy")
        if type(self.decision_reason_code) is not RecoveryDecisionReasonCode:
            raise TypeError("decision_reason_code must be RecoveryDecisionReasonCode")
        object.__setattr__(
            self,
            "evidence_refs",
            freeze_tuple(self.evidence_refs, label="evidence_refs"),
        )
        for ref in self.evidence_refs:
            validate_problem_reference(ref)
        object.__setattr__(
            self,
            "decided_at",
            require_aware_utc(self.decided_at, label="decided_at"),
        )
        object.__setattr__(
            self,
            "source_ref",
            require_non_empty_text(self.source_ref, label="source_ref"),
        )
        object.__setattr__(
            self,
            "decision_policy_version",
            require_non_empty_text(
                self.decision_policy_version,
                label="decision_policy_version",
            ),
        )
        if self.retry_after is not None:
            object.__setattr__(
                self,
                "retry_after",
                require_aware_utc(self.retry_after, label="retry_after"),
            )
        if self.max_attempts is not None and self.max_attempts < 0:
            raise ValueError("max_attempts must be non-negative")
        if self.strategy is RecoveryStrategy.RETRY:
            if self.max_attempts is None or self.max_attempts <= 0:
                raise ValueError("RETRY strategy requires max_attempts > 0")
        if self.resume_target_ref is not None:
            object.__setattr__(
                self,
                "resume_target_ref",
                require_non_empty_text(
                    self.resume_target_ref,
                    label="resume_target_ref",
                ),
            )
        if self.human_decision_ref is not None:
            validate_human_pending_reference(self.human_decision_ref)
        if self.dependency_ref is not None:
            validate_external_dependency_reference(self.dependency_ref)
        if self.strategy is RecoveryStrategy.STOP:
            if self.retry_after is not None or self.max_attempts is not None:
                raise ValueError("STOP strategy must not carry retry fields")
        if self.strategy is RecoveryStrategy.QUARANTINE:
            if self.retry_after is not None:
                raise ValueError("QUARANTINE strategy must not carry retry_after")


@dataclass(frozen=True, slots=True)
class WorkerRecoveryDecisionContext:
    """Profile and policy context for bounded recovery decisions."""

    capability_acquisition_allowed: bool = False
    decision_policy_version: str = DECISION_POLICY_VERSION
    max_retry_attempts: int | None = None


@dataclass(frozen=True, slots=True)
class WorkerRecoveryDecisionResult:
    """Classification + decision bundle with typed disposition."""

    disposition: ObstacleClassificationDisposition
    classification: WorkerObstacleClassification | None
    decision: WorkerRecoveryDecision | None


class WorkerObstacleClassifier(Protocol):
    """Optional domain classifier — canonical safety mappings are not overrideable."""

    classifier_id: str

    def classify(
        self,
        evidence: WorkerObstacleEvidence,
        *,
        classified_at: datetime,
    ) -> WorkerObstacleClassification | None:
        ...


def is_safety_critical_obstacle_kind(kind: WorkerObstacleKind) -> bool:
    return kind in {
        WorkerObstacleKind.POLICY_DENIED,
        WorkerObstacleKind.CREDENTIAL_UNAVAILABLE,
        WorkerObstacleKind.SUSPICIOUS_OR_UNSAFE,
    }


def policy_action_is_obstacle(action: PolicyAction) -> bool:
    return action is not PolicyAction.ALLOW
