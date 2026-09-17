# © Artur Czarnecki. All rights reserved.

"""GR-7-A8 — provider invocation reliability evidence chain (provider-neutral)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
    UnknownUncertaintyPosture,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDecision,
    ProviderInvocationRecoveryDispatchState,
    ProviderInvocationRecoveryReason,
    ProviderInvocationRecoveryRequest,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence import (
    ProviderInvocationReliabilityFact,
    ProviderInvocationReliabilityTracePhase,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityReason,
    ExternalEffectRepeatEligibilityResult,
    ExternalEffectRepeatEligibilityVerdict,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryExecutionDisposition,
    ProviderInvocationRecoveryExecutionPorts,
    ProviderInvocationRecoveryExecutionResult,
    decide_provider_invocation_recovery,
    execute_provider_invocation_recovery,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_reliability_evidence import (
    emit_provider_invocation_reliability_fact,
    project_crash_ambiguity,
    project_intent_persisted,
    project_outcome_persisted,
    project_recovery_execution,
    project_repeat_attempt_linked,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery_execution_validation import (
    ProviderInvocationRecoveryExecutionBlockReason,
)

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 9, 17, 10, 0, 0, tzinfo=UTC)
_TENANT = "tenant-gr7a8"


@dataclass
class _RecordingObserver:
    facts: list[ProviderInvocationReliabilityFact] = field(default_factory=list)

    def observe_provider_invocation_reliability_fact(
        self,
        fact: ProviderInvocationReliabilityFact,
    ) -> None:
        self.facts.append(fact)


class _FailingObserver:
    def observe_provider_invocation_reliability_fact(
        self,
        fact: ProviderInvocationReliabilityFact,
    ) -> None:
        raise RuntimeError("telemetry sink unavailable")


def _inv(**kwargs: object) -> ProviderInvocation:
    base = {
        "invocation_id": "inv-original",
        "provider_id": "prov-generic",
        "operation": "generic.effect.apply",
        "task_id": "task-1",
        "run_id": "run-1",
        "idempotency_key": "idem-logical-1",
        "request_digest": "digest-a",
        "started_at": _T0,
    }
    base.update(kwargs)
    return ProviderInvocation.model_validate(base)


def _outcome(**kwargs: object) -> ProviderInvocationOutcome:
    base = {
        "invocation_id": "inv-original",
        "status": ProviderInvocationStatus.FAILED,
        "completed_at": _T0,
    }
    base.update(kwargs)
    return ProviderInvocationOutcome.model_validate(base)


def _contract() -> ExternalEffectContract:
    return ExternalEffectContract(
        contract_id="generic.effect.apply.v1",
        operation_key="generic.effect.apply",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("probe.read",),
    )


def _eligibility() -> ExternalEffectRepeatEligibilityResult:
    return ExternalEffectRepeatEligibilityResult(
        verdict=ExternalEffectRepeatEligibilityVerdict.ELIGIBLE,
        reason=ExternalEffectRepeatEligibilityReason.ALLOWED_IDEMPOTENT_REPEAT,
        invocation_id="inv-original",
        idempotency_key="idem-logical-1",
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_OR_IDEMPOTENT_REPEAT,
        policy_id="repeat-policy-test",
    )


def test_intent_and_outcome_phases_distinguish_unknown() -> None:
    observer = _RecordingObserver()
    inv = _inv()
    emit_provider_invocation_reliability_fact(
        project_intent_persisted(
            invocation=inv,
            tenant_id=_TENANT,
            effect_contract_id=_contract().contract_id,
            recorded_at=_T0,
        ),
        observer,
    )
    emit_provider_invocation_reliability_fact(
        project_outcome_persisted(
            invocation=inv,
            outcome=_outcome(status=ProviderInvocationStatus.UNKNOWN),
            tenant_id=_TENANT,
            effect_contract_id=_contract().contract_id,
            recorded_at=_T0,
        ),
        observer,
    )
    phases = [f.phase for f in observer.facts]
    assert phases == [
        ProviderInvocationReliabilityTracePhase.INTENT_PERSISTED,
        ProviderInvocationReliabilityTracePhase.UNKNOWN_ADMITTED,
    ]
    assert observer.facts[1].invocation_status is ProviderInvocationStatus.UNKNOWN


def test_crash_ambiguity_distinct_from_unknown_outcome() -> None:
    observer = _RecordingObserver()
    emit_provider_invocation_reliability_fact(
        project_crash_ambiguity(
            invocation=_inv(),
            tenant_id=_TENANT,
            effect_contract_id=_contract().contract_id,
            recorded_at=_T0,
        ),
        observer,
    )
    assert observer.facts[0].phase is (
        ProviderInvocationReliabilityTracePhase.CRASH_AMBIGUITY_ADMITTED
    )
    assert (
        observer.facts[0].dispatch_state
        is ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY
    )


def test_recovery_decision_trace_includes_repeat_eligibility() -> None:
    observer = _RecordingObserver()
    request = ProviderInvocationRecoveryRequest(
        invocation=_inv(),
        outcome=_outcome(),
        effect_contract=_contract(),
        repeat_eligibility=_eligibility(),
        repeat_execution_supported=False,
    )
    decide_provider_invocation_recovery(
        request,
        evidence_observer=observer,
        tenant_id=_TENANT,
        recorded_at=_T0,
    )
    kinds = {f.phase for f in observer.facts}
    assert ProviderInvocationReliabilityTracePhase.REPEAT_ELIGIBILITY_EVALUATED in kinds
    assert ProviderInvocationReliabilityTracePhase.RECOVERY_DECIDED in kinds
    eligibility_fact = next(
        f
        for f in observer.facts
        if f.phase is ProviderInvocationReliabilityTracePhase.REPEAT_ELIGIBILITY_EVALUATED
    )
    assert eligibility_fact.repeat_policy_id == "repeat-policy-test"
    assert eligibility_fact.repeat_eligibility_verdict is (
        ExternalEffectRepeatEligibilityVerdict.ELIGIBLE
    )


def test_recovery_execution_blocked_trace_zero_mutations() -> None:
    observer = _RecordingObserver()
    inv = _inv()
    contract = _contract()
    decision = ProviderInvocationRecoveryDecision(
        action=ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,
        reason=ProviderInvocationRecoveryReason.POLICY_SELECTED,
        invocation_id="inv-original",
        policy_id="recovery-policy",
    )
    execution = ProviderInvocationRecoveryExecutionResult(
        decision=decision,
        execution_attempted=True,
        disposition=ProviderInvocationRecoveryExecutionDisposition.BLOCKED,
        provider_mutation_count=0,
        block_reason=ProviderInvocationRecoveryExecutionBlockReason.DECISION_ACTION_MISMATCH,
    )
    emit_provider_invocation_reliability_fact(
        project_recovery_execution(
            invocation=inv,
            tenant_id=_TENANT,
            effect_contract_id=contract.contract_id,
            execution=execution,
            recorded_at=_T0,
        ),
        observer,
    )
    execution_fact = observer.facts[0]
    assert execution_fact.provider_mutation_count == 0
    assert (
        execution_fact.recovery_block_reason
        == ProviderInvocationRecoveryExecutionBlockReason.DECISION_ACTION_MISMATCH.value
    )


def test_repeat_attempt_linked_preserves_logical_idempotency() -> None:
    observer = _RecordingObserver()
    inv = _inv()
    emit_provider_invocation_reliability_fact(
        project_repeat_attempt_linked(
            invocation=inv,
            tenant_id=_TENANT,
            effect_contract_id=_contract().contract_id,
            repeat_invocation_id="inv-repeat-2",
            recorded_at=_T0,
        ),
        observer,
    )
    fact = observer.facts[0]
    assert fact.repeat_invocation_id == "inv-repeat-2"
    assert fact.correlation.invocation_id == "inv-original"
    assert fact.correlation.idempotency_key == "idem-logical-1"


def test_observer_failure_does_not_raise() -> None:
    emit_provider_invocation_reliability_fact(
        project_intent_persisted(
            invocation=_inv(),
            tenant_id=_TENANT,
            effect_contract_id=_contract().contract_id,
            recorded_at=_T0,
        ),
        _FailingObserver(),
    )


def test_succeeded_path_terminal_no_execution_mutation() -> None:
    observer = _RecordingObserver()
    contract = ExternalEffectContract(
        contract_id="external_work.create_work.v1",
        operation_key="external_work.create_work",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
    )
    request = ProviderInvocationRecoveryRequest(
        invocation=_inv(operation="external_work.create_work"),
        outcome=_outcome(status=ProviderInvocationStatus.SUCCEEDED),
        effect_contract=contract,
    )
    decision = decide_provider_invocation_recovery(
        request,
        evidence_observer=observer,
        tenant_id=_TENANT,
        recorded_at=_T0,
    )
    assert decision.action is ProviderInvocationRecoveryAction.NO_ACTION
    execution = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(),
        evidence_observer=observer,
        tenant_id=_TENANT,
        recorded_at=_T0,
    )
    assert execution.disposition is ProviderInvocationRecoveryExecutionDisposition.NOT_ATTEMPTED
    assert execution.provider_mutation_count == 0
