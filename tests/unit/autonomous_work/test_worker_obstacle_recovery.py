# © Artur Czarnecki. All rights reserved.

"""AW-6A — canonical obstacle taxonomy and recovery decision tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.autonomous_work.obstacle_recovery_ports import StaticCapabilityAcquisitionPolicy
from intergrax.autonomous_work.worker_obstacle_classifier import CanonicalWorkerObstacleClassifier
from intergrax.autonomous_work.worker_recovery_decision_service import (
    WorkerRecoveryDecisionService,
)
from intergrax.contracts.autonomous_work.obstacle_recovery import (
    DECISION_POLICY_VERSION,
    ObstacleClassificationDisposition,
    ObstacleClassificationReasonCode,
    RecoveryDecisionReasonCode,
    RecoveryStrategy,
    WorkerObstacleClassification,
    WorkerObstacleEvidence,
    WorkerObstacleKind,
    WorkerObstacleSourceKind,
    WorkerRecoveryDecisionContext,
    derive_recovery_decision_id,
    derive_worker_obstacle_id,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    CapabilityProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import (
    ExternalDependencyReference,
    ProblemReference,
)
from intergrax.contracts.policy_action import PolicyAction
from intergrax.contracts.resilience_policy import FailureClass, FailureResponse, ResiliencePolicy
from intergrax.contracts.runtime_policy import PolicyDecision
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=_UTC)
_EVIDENCE_REF = ProblemReference("problem/evidence/terminal-failure-1")
_WORKER_ID = contract_suite.mint_worker_instance_id()
_CAPABILITY_PROFILE = CapabilityProfileRef(
    profile_id="cap/default",
    version=initial_profile_version(),
)


def _service(**kwargs) -> WorkerRecoveryDecisionService:
    return WorkerRecoveryDecisionService(**kwargs)


def _evidence(**overrides) -> WorkerObstacleEvidence:
    base = WorkerObstacleEvidence(
        worker_instance_id=_WORKER_ID,
        source_kind=WorkerObstacleSourceKind.EXECUTION_FAILURE,
        source_ref="execution/terminal/failed-1",
        occurrence_identity="terminal-evidence-1",
        observed_at=_NOW,
        problem_evidence_refs=(_EVIDENCE_REF,),
    )
    if not overrides:
        return base
    return replace(base, **overrides)


def _decide(evidence: WorkerObstacleEvidence, **context_kwargs):
    return _service().decide(
        evidence,
        context=WorkerRecoveryDecisionContext(**context_kwargs),
        decided_at=_NOW,
    )


def test_transient_failure_maps_to_bounded_retry() -> None:
    result = _decide(
        _evidence(
            runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
            failure_class=FailureClass.RUNTIME_ERROR,
        ),
        max_retry_attempts=3,
    )
    assert result.disposition is ObstacleClassificationDisposition.CLASSIFIED
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.TRANSIENT_FAILURE
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.RETRY
    assert result.decision.max_attempts == 3
    assert (
        result.decision.decision_reason_code
        is RecoveryDecisionReasonCode.TRANSIENT_RETRY_BOUNDED
    )


def test_transient_without_bounded_policy_escalates() -> None:
    result = _decide(
        _evidence(runtime_error_code=RuntimeErrorCode.TIMEOUT.value),
        max_retry_attempts=0,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE
    assert (
        result.decision.decision_reason_code
        is RecoveryDecisionReasonCode.TRANSIENT_RETRY_UNBOUNDED_ESCALATE
    )


def test_transient_without_retry_policy_or_context_escalates() -> None:
    result = _service().decide(
        _evidence(runtime_error_code=RuntimeErrorCode.TIMEOUT.value),
        decided_at=_NOW,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE
    assert result.decision.max_attempts is None
    assert (
        result.decision.decision_reason_code
        is RecoveryDecisionReasonCode.TRANSIENT_RETRY_UNBOUNDED_ESCALATE
    )


def test_transient_explicit_context_retry_limit() -> None:
    result = _decide(
        _evidence(runtime_error_code=RuntimeErrorCode.TIMEOUT.value),
        max_retry_attempts=2,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.RETRY
    assert result.decision.max_attempts == 2


def test_dependency_unavailable_maps_to_wait() -> None:
    result = _decide(
        _evidence(
            source_kind=WorkerObstacleSourceKind.DEPENDENCY_STATUS,
            dependency_unavailable=True,
            dependency_ref=ExternalDependencyReference("external/vendor-api"),
        ),
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.DEPENDENCY_UNAVAILABLE
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.WAIT
    assert result.decision.decision_reason_code is RecoveryDecisionReasonCode.DEPENDENCY_WAIT


def test_rate_limited_throttle_without_retry_after() -> None:
    result = _decide(_evidence(rate_limited=True))
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.RATE_LIMITED
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.THROTTLE


def test_rate_limited_wait_preserves_retry_after() -> None:
    retry_after = _NOW + timedelta(seconds=30)
    result = _decide(_evidence(rate_limited=True, retry_after=retry_after))
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.WAIT
    assert result.decision.retry_after == retry_after


def test_credential_unavailable_escalates() -> None:
    result = _decide(
        _evidence(
            credential_ref="credential/api-key-revoked",
            runtime_error_code=RuntimeErrorCode.PERMISSION_ERROR.value,
        ),
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.CREDENTIAL_UNAVAILABLE
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE
    assert result.decision.decision_reason_code is RecoveryDecisionReasonCode.CREDENTIAL_ESCALATE


def test_policy_deny_maps_to_stop() -> None:
    result = _decide(
        _evidence(
            policy_decision=PolicyDecision(action=PolicyAction.DENY, reason="forbidden"),
            runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
        ),
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.POLICY_DENIED
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.STOP
    assert result.decision.retry_after is None
    assert result.decision.max_attempts is None


def test_policy_require_human_maps_to_request_human_decision() -> None:
    result = _decide(
        _evidence(
            source_kind=WorkerObstacleSourceKind.POLICY_DECISION,
            policy_decision=PolicyDecision(action=PolicyAction.REQUIRE_HUMAN),
        ),
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.HUMAN_DECISION_REQUIRED
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.REQUEST_HUMAN_DECISION


def test_business_ambiguity_maps_to_human_decision() -> None:
    result = _decide(
        _evidence(
            source_kind=WorkerObstacleSourceKind.BUSINESS_DECISION,
            business_ambiguity=True,
        ),
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.BUSINESS_AMBIGUITY
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.REQUEST_HUMAN_DECISION


def test_alternative_path_maps_to_replan_when_explicit() -> None:
    result = _decide(
        _evidence(
            alternative_path_ref="capability/alternate-tool-approved",
            failure_class=FailureClass.QUALITY_ERROR,
        ),
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.ALTERNATIVE_PATH_AVAILABLE
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.REPLAN


def test_quality_error_without_alternative_escalates_not_replan() -> None:
    result = _decide(_evidence(failure_class=FailureClass.QUALITY_ERROR))
    assert result.classification is not None
    assert result.classification.obstacle_kind is not WorkerObstacleKind.ALTERNATIVE_PATH_AVAILABLE
    assert result.decision is not None
    assert result.decision.strategy is not RecoveryStrategy.REPLAN
    assert result.decision.strategy is RecoveryStrategy.ESCALATE


def test_user_error_without_business_ambiguity_does_not_request_human() -> None:
    result = _decide(_evidence(failure_class=FailureClass.USER_ERROR))
    assert result.classification is not None
    assert result.classification.obstacle_kind is not WorkerObstacleKind.BUSINESS_AMBIGUITY
    assert result.decision is not None
    assert result.decision.strategy is not RecoveryStrategy.REQUEST_HUMAN_DECISION


def test_schema_drift_maps_to_adapt_integration() -> None:
    result = _decide(_evidence(schema_drift_detected=True))
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.SCHEMA_OR_API_DRIFT
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ADAPT_INTEGRATION


def test_missing_capability_acquire_when_allowed() -> None:
    service = _service(
        capability_policy=StaticCapabilityAcquisitionPolicy(allowed=True),
    )
    result = service.decide(
        _evidence(
            source_kind=WorkerObstacleSourceKind.CAPABILITY_RESOLUTION,
            capability_missing_ref="tool/pdf-parser",
            capability_profile_ref=_CAPABILITY_PROFILE,
        ),
        context=WorkerRecoveryDecisionContext(),
        decided_at=_NOW,
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.CAPABILITY_MISSING
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ACQUIRE_CAPABILITY


def test_missing_capability_escalate_when_not_allowed() -> None:
    result = _decide(
        _evidence(
            source_kind=WorkerObstacleSourceKind.CAPABILITY_RESOLUTION,
            capability_missing_ref="tool/pdf-parser",
        ),
        capability_acquisition_allowed=False,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE


def test_suspicious_maps_to_quarantine() -> None:
    result = _decide(_evidence(suspicious_or_unsafe=True))
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.SUSPICIOUS_OR_UNSAFE
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.QUARANTINE


def test_unknown_escalates_not_retry() -> None:
    result = _decide(_evidence())
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.UNKNOWN
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE
    assert result.decision.strategy is not RecoveryStrategy.RETRY


def test_policy_overrides_retry_signal() -> None:
    result = _decide(
        _evidence(
            policy_decision=PolicyDecision(action=PolicyAction.DENY),
            runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
            failure_class=FailureClass.RUNTIME_ERROR,
        ),
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.STOP


def test_credential_overrides_capability_signal() -> None:
    result = _decide(
        _evidence(
            credential_ref="credential/revoked",
            capability_missing_ref="tool/parser",
            source_kind=WorkerObstacleSourceKind.CAPABILITY_RESOLUTION,
        ),
        capability_acquisition_allowed=True,
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.CREDENTIAL_UNAVAILABLE
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE


def test_conflicting_domain_classifiers_escalate() -> None:
    class _ClassifierA:
        classifier_id = "domain.a"

        def classify(self, evidence, *, classified_at):
            return WorkerObstacleClassification(
                obstacle_id=derive_worker_obstacle_id(evidence),
                obstacle_kind=WorkerObstacleKind.TRANSIENT_FAILURE,
                classifier_id=self.classifier_id,
                reason_code=ObstacleClassificationReasonCode.TRANSIENT_RUNTIME_FAILURE,
                evidence_refs=evidence.problem_evidence_refs,
                classified_at=classified_at,
            )

    class _ClassifierB:
        classifier_id = "domain.b"

        def classify(self, evidence, *, classified_at):
            return WorkerObstacleClassification(
                obstacle_id=derive_worker_obstacle_id(evidence),
                obstacle_kind=WorkerObstacleKind.DEPENDENCY_UNAVAILABLE,
                classifier_id=self.classifier_id,
                reason_code=ObstacleClassificationReasonCode.DEPENDENCY_UNAVAILABLE,
                evidence_refs=evidence.problem_evidence_refs,
                classified_at=classified_at,
            )

    service = _service(domain_classifiers=[_ClassifierA(), _ClassifierB()])
    result = service.decide(_evidence(runtime_error_code=RuntimeErrorCode.TIMEOUT.value))
    assert result.disposition is ObstacleClassificationDisposition.CONFLICT
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE


def test_determinism_same_evidence_same_decision() -> None:
    evidence = _evidence(
        runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
        failure_class=FailureClass.RUNTIME_ERROR,
    )
    context = WorkerRecoveryDecisionContext(max_retry_attempts=2)
    first = _service().decide(evidence, context=context, decided_at=_NOW)
    second = _service().decide(evidence, context=context, decided_at=_NOW)
    assert first.classification is not None and second.classification is not None
    assert first.decision is not None and second.decision is not None
    assert (
        first.classification.obstacle_kind
        == second.classification.obstacle_kind
    )
    assert first.decision.strategy == second.decision.strategy
    assert first.decision.decision_reason_code == second.decision.decision_reason_code
    assert first.decision.decision_id == second.decision.decision_id


def test_stable_obstacle_and_decision_identity() -> None:
    evidence = _evidence(runtime_error_code=RuntimeErrorCode.TIMEOUT.value)
    obstacle_id = derive_worker_obstacle_id(evidence)
    decision_id = derive_recovery_decision_id(obstacle_id)
    result = _decide(evidence)
    assert result.classification is not None
    assert result.classification.obstacle_id == obstacle_id
    assert result.decision is not None
    assert result.decision.decision_id == decision_id
    assert result.decision.decision_policy_version == DECISION_POLICY_VERSION


def test_reuses_resilience_policy_max_attempts() -> None:
    policy = ResiliencePolicy(max_attempts=5)
    service = _service(resilience_policy=policy)
    result = service.decide(
        _evidence(
            runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
            failure_class=FailureClass.RUNTIME_ERROR,
        ),
        decided_at=_NOW,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.RETRY
    assert result.decision.max_attempts == 5


def test_resilience_policy_non_retry_response_escalates() -> None:
    policy = ResiliencePolicy(
        on_runtime_error=FailureResponse.FAIL,
        max_attempts=5,
    )
    service = _service(resilience_policy=policy)
    result = service.decide(
        _evidence(
            runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
            failure_class=FailureClass.RUNTIME_ERROR,
        ),
        decided_at=_NOW,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE


def _transient_evidence():
    return _evidence(
        runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
        failure_class=FailureClass.RUNTIME_ERROR,
    )


def test_resilience_policy_fail_overrides_context_retry_limit() -> None:
    policy = ResiliencePolicy(
        on_runtime_error=FailureResponse.FAIL,
        max_attempts=5,
    )
    service = _service(resilience_policy=policy)
    result = service.decide(
        _transient_evidence(),
        context=WorkerRecoveryDecisionContext(max_retry_attempts=2),
        decided_at=_NOW,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE
    assert result.decision.strategy is not RecoveryStrategy.RETRY


def test_resilience_policy_escalate_overrides_context_retry_limit() -> None:
    policy = ResiliencePolicy(
        on_runtime_error=FailureResponse.ESCALATE,
        max_attempts=5,
    )
    service = _service(resilience_policy=policy)
    result = service.decide(
        _transient_evidence(),
        context=WorkerRecoveryDecisionContext(max_retry_attempts=2),
        decided_at=_NOW,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE


def test_resilience_policy_retry_narrowed_by_context_limit() -> None:
    policy = ResiliencePolicy(max_attempts=5)
    service = _service(resilience_policy=policy)
    result = service.decide(
        _transient_evidence(),
        context=WorkerRecoveryDecisionContext(max_retry_attempts=2),
        decided_at=_NOW,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.RETRY
    assert result.decision.max_attempts == 2


def test_resilience_policy_retry_tighter_than_context_limit() -> None:
    policy = ResiliencePolicy(max_attempts=2)
    service = _service(resilience_policy=policy)
    result = service.decide(
        _transient_evidence(),
        context=WorkerRecoveryDecisionContext(max_retry_attempts=5),
        decided_at=_NOW,
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.RETRY
    assert result.decision.max_attempts == 2


def test_canonical_classifier_is_deterministic_first() -> None:
    classifier = CanonicalWorkerObstacleClassifier()
    evidence = _evidence(
        policy_decision=PolicyDecision(action=PolicyAction.DENY),
        runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
    )
    classification = classifier.classify(evidence, classified_at=_NOW)
    assert classification.obstacle_kind is WorkerObstacleKind.POLICY_DENIED
    assert classification.reason_code is ObstacleClassificationReasonCode.POLICY_DENIED


def test_canonical_unknown_refined_by_domain_classifier() -> None:
    class _SchemaDriftClassifier:
        classifier_id = "domain.schema"

        def classify(self, evidence, *, classified_at):
            return WorkerObstacleClassification(
                obstacle_id=derive_worker_obstacle_id(evidence),
                obstacle_kind=WorkerObstacleKind.SCHEMA_OR_API_DRIFT,
                classifier_id=self.classifier_id,
                reason_code=ObstacleClassificationReasonCode.SCHEMA_OR_API_DRIFT,
                evidence_refs=evidence.problem_evidence_refs,
                classified_at=classified_at,
            )

    service = _service(domain_classifiers=[_SchemaDriftClassifier()])
    result = service.decide(_evidence())
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.SCHEMA_OR_API_DRIFT
    assert result.classification.classifier_id == "domain.schema"


def test_canonical_unknown_agreeing_domain_classifiers_use_plugin() -> None:
    class _SchemaDriftA:
        classifier_id = "domain.schema.a"

        def classify(self, evidence, *, classified_at):
            return WorkerObstacleClassification(
                obstacle_id=derive_worker_obstacle_id(evidence),
                obstacle_kind=WorkerObstacleKind.SCHEMA_OR_API_DRIFT,
                classifier_id=self.classifier_id,
                reason_code=ObstacleClassificationReasonCode.SCHEMA_OR_API_DRIFT,
                evidence_refs=evidence.problem_evidence_refs,
                classified_at=classified_at,
            )

    class _SchemaDriftB:
        classifier_id = "domain.schema.b"

        def classify(self, evidence, *, classified_at):
            return WorkerObstacleClassification(
                obstacle_id=derive_worker_obstacle_id(evidence),
                obstacle_kind=WorkerObstacleKind.SCHEMA_OR_API_DRIFT,
                classifier_id=self.classifier_id,
                reason_code=ObstacleClassificationReasonCode.SCHEMA_OR_API_DRIFT,
                evidence_refs=evidence.problem_evidence_refs,
                classified_at=classified_at,
            )

    service = _service(domain_classifiers=[_SchemaDriftA(), _SchemaDriftB()])
    result = service.decide(_evidence())
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.SCHEMA_OR_API_DRIFT


def test_canonical_policy_denied_domain_transient_never_retries() -> None:
    class _TransientClassifier:
        classifier_id = "domain.transient"

        def classify(self, evidence, *, classified_at):
            return WorkerObstacleClassification(
                obstacle_id=derive_worker_obstacle_id(evidence),
                obstacle_kind=WorkerObstacleKind.TRANSIENT_FAILURE,
                classifier_id=self.classifier_id,
                reason_code=ObstacleClassificationReasonCode.TRANSIENT_RUNTIME_FAILURE,
                evidence_refs=evidence.problem_evidence_refs,
                classified_at=classified_at,
            )

    service = _service(domain_classifiers=[_TransientClassifier()])
    result = service.decide(
        _evidence(
            policy_decision=PolicyDecision(action=PolicyAction.DENY),
            runtime_error_code=RuntimeErrorCode.TIMEOUT.value,
        ),
        context=WorkerRecoveryDecisionContext(max_retry_attempts=3),
        decided_at=_NOW,
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.POLICY_DENIED
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.STOP


def test_canonical_credential_domain_capability_never_acquires() -> None:
    class _CapabilityClassifier:
        classifier_id = "domain.capability"

        def classify(self, evidence, *, classified_at):
            return WorkerObstacleClassification(
                obstacle_id=derive_worker_obstacle_id(evidence),
                obstacle_kind=WorkerObstacleKind.CAPABILITY_MISSING,
                classifier_id=self.classifier_id,
                reason_code=ObstacleClassificationReasonCode.CAPABILITY_MISSING,
                evidence_refs=evidence.problem_evidence_refs,
                classified_at=classified_at,
            )

    service = _service(
        domain_classifiers=[_CapabilityClassifier()],
        capability_policy=StaticCapabilityAcquisitionPolicy(allowed=True),
    )
    result = service.decide(
        _evidence(
            credential_ref="credential/revoked",
            capability_missing_ref="tool/parser",
            source_kind=WorkerObstacleSourceKind.CAPABILITY_RESOLUTION,
            capability_profile_ref=_CAPABILITY_PROFILE,
        ),
        decided_at=_NOW,
    )
    assert result.classification is not None
    assert result.classification.obstacle_kind is WorkerObstacleKind.CREDENTIAL_UNAVAILABLE
    assert result.decision is not None
    assert result.decision.strategy is not RecoveryStrategy.ACQUIRE_CAPABILITY


def test_retry_contract_rejects_missing_max_attempts() -> None:
    from intergrax.contracts.autonomous_work.obstacle_recovery import WorkerRecoveryDecision

    with pytest.raises(ValueError, match="max_attempts"):
        WorkerRecoveryDecision(
            decision_id="decision-1",
            obstacle_id="obstacle-1",
            obstacle_kind=WorkerObstacleKind.TRANSIENT_FAILURE,
            strategy=RecoveryStrategy.RETRY,
            decision_reason_code=RecoveryDecisionReasonCode.TRANSIENT_RETRY_BOUNDED,
            evidence_refs=(_EVIDENCE_REF,),
            decided_at=_NOW,
            source_ref="execution/terminal/failed-1",
        )


def test_retry_contract_rejects_zero_max_attempts() -> None:
    from intergrax.contracts.autonomous_work.obstacle_recovery import WorkerRecoveryDecision

    with pytest.raises(ValueError, match="max_attempts"):
        WorkerRecoveryDecision(
            decision_id="decision-1",
            obstacle_id="obstacle-1",
            obstacle_kind=WorkerObstacleKind.TRANSIENT_FAILURE,
            strategy=RecoveryStrategy.RETRY,
            decision_reason_code=RecoveryDecisionReasonCode.TRANSIENT_RETRY_BOUNDED,
            evidence_refs=(_EVIDENCE_REF,),
            decided_at=_NOW,
            source_ref="execution/terminal/failed-1",
            max_attempts=0,
        )


def test_dependency_without_resume_semantics_escalates() -> None:
    result = _decide(
        _evidence(
            source_kind=WorkerObstacleSourceKind.DEPENDENCY_STATUS,
            dependency_unavailable=True,
        ),
    )
    assert result.decision is not None
    assert result.decision.strategy is RecoveryStrategy.ESCALATE
