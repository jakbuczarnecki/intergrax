# © Artur Czarnecki. All rights reserved.

"""GR-7-A7-R1 — recovery decision→execution integrity and governed repeat."""

from __future__ import annotations

from datetime import UTC, datetime
import pytest

from applications.governed_contractor_application.host.governed_external_work_recovery_ports import (
    GovernedExternalWorkProviderRecoveryHitlPort,
    GovernedExternalWorkProviderRecoveryRepeatPort,
)
from applications.governed_contractor_application.host.provider_invocation_recovery import (
    GovernedExternalWorkProviderRecovery,
)
from applications.governed_contractor_application.host.provider_invocation_reconciliation import (
    GovernedExternalWorkProviderReconciliation,
)
from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    _FAIL_IDEMP,
    _PRINCIPAL,
    _TENANT,
    _build_runtime,
    _create_meta,
    _create_step,
)
from applications.governed_contractor_application.tests.host.test_gr7_a7_provider_recovery import (
    _AllowRepeatPolicy,
    _RecordingRepeatPort,
    _invocation,
    _outcome,
    _repeat_eligibility,
)
from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.stores import InMemoryContinuationStateStore
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
    UnknownUncertaintyPosture,
    evaluate_unknown_uncertainty_posture,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationReason,
    ProviderInvocationReconciliationResult,
    ProviderInvocationReconciliationVerdict,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDecision,
    ProviderInvocationRecoveryDispatchState,
    ProviderInvocationRecoveryReason,
    ProviderInvocationRecoveryRequest,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    decide_provider_invocation_recovery,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityReason,
    ExternalEffectRepeatEligibilityResult,
    ExternalEffectRepeatEligibilityVerdict,
)
from intergrax.contracts.external_work import ExternalWorkErrorCode
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryExecutionBlockReason,
    ProviderInvocationRecoveryExecutionDisposition,
    ProviderInvocationRecoveryExecutionPorts,
    execute_provider_invocation_recovery,
)
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=UTC)
_CREATE_CONTRACT = external_work_effect_contract_for_action(
    ACTION_CREATE_EXTERNAL_WORK,
    quote_first_partner_capability_fixture(),
)


def _eligible_repeat_request(
    *,
    invocation_id: str = "inv-r1",
    idempotency_key: str = "idem-r1",
) -> ProviderInvocationRecoveryRequest:
    inv = _invocation(invocation_id=invocation_id, idempotency_key=idempotency_key)
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    return ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
    )


def _repeat_decision_for(inv: object, key: str | None) -> ProviderInvocationRecoveryDecision:
    from intergrax.contracts.provider_invocation import ProviderInvocation

    assert isinstance(inv, ProviderInvocation)
    return ProviderInvocationRecoveryDecision(
        action=ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,
        reason=ProviderInvocationRecoveryReason.POLICY_SELECTED,
        invocation_id=inv.invocation_id,
        idempotency_key=key,
    )


def _execute_repeat_blocked(
    request: ProviderInvocationRecoveryRequest,
    decision: ProviderInvocationRecoveryDecision,
) -> None:
    repeat_port = _RecordingRepeatPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert repeat_port.calls == 0
    assert result.provider_mutation_count == 0
    assert result.disposition is ProviderInvocationRecoveryExecutionDisposition.BLOCKED


def test_foreign_decision_invocation_id_blocked() -> None:
    request = _eligible_repeat_request()
    inv = request.invocation
    assert inv is not None
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    decision = decision.model_copy(update={"invocation_id": "inv-other"})
    _execute_repeat_blocked(request, decision)


def test_foreign_decision_idempotency_key_blocked() -> None:
    request = _eligible_repeat_request()
    inv = request.invocation
    assert inv is not None
    decision = _repeat_decision_for(inv, "wrong-key")
    _execute_repeat_blocked(request, decision)


def test_repeat_decision_not_eligible_blocked() -> None:
    request = _eligible_repeat_request()
    inv = request.invocation
    assert inv is not None
    outcome = request.outcome
    assert outcome is not None
    request = request.model_copy(
        update={
            "repeat_eligibility": ExternalEffectRepeatEligibilityResult(
                verdict=ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED,
                reason=ExternalEffectRepeatEligibilityReason.DENIED_POLICY,
                invocation_id=inv.invocation_id,
                idempotency_key=inv.idempotency_key,
            ),
        },
    )
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_repeat_decision_eligibility_missing_blocked() -> None:
    request = _eligible_repeat_request().model_copy(update={"repeat_eligibility": None})
    inv = request.invocation
    assert inv is not None
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_repeat_eligibility_foreign_invocation_blocked() -> None:
    request = _eligible_repeat_request()
    inv = request.invocation
    assert inv is not None
    outcome = request.outcome
    assert outcome is not None
    request = request.model_copy(
        update={
            "repeat_eligibility": _repeat_eligibility(inv, outcome).model_copy(
                update={"invocation_id": "foreign-inv"},
            ),
        },
    )
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_repeat_eligibility_foreign_idempotency_blocked() -> None:
    request = _eligible_repeat_request()
    inv = request.invocation
    assert inv is not None
    outcome = request.outcome
    assert outcome is not None
    request = request.model_copy(
        update={
            "repeat_eligibility": _repeat_eligibility(inv, outcome).model_copy(
                update={"idempotency_key": "foreign-key"},
            ),
        },
    )
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_crash_ambiguity_forged_repeat_blocked() -> None:
    inv = _invocation()
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=None,
        dispatch_state=ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY,
        effect_contract=_CREATE_CONTRACT,
    )
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_succeeded_forged_repeat_blocked() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.SUCCEEDED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
    )
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_reconciled_success_forged_repeat_blocked() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    reconciliation = ProviderInvocationReconciliationResult(
        verdict=ProviderInvocationReconciliationVerdict.CONFIRMED_SUCCEEDED,
        reason=ProviderInvocationReconciliationReason.PROBE_EXECUTED,
        invocation_id=inv.invocation_id,
    )
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        reconciliation=reconciliation,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
    )
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_reconcile_only_posture_forged_repeat_blocked() -> None:
    contract = ExternalEffectContract(
        contract_id="test.reconcile_only.r1",
        operation_key="external_work.accept_quote",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("get_work",),
    )
    assert evaluate_unknown_uncertainty_posture(contract) is UnknownUncertaintyPosture.RECONCILE_ONLY
    inv = _invocation(invocation_id="inv-accept-r1").model_copy(
        update={"operation": "external_work.accept_quote"},
    )
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=contract,
        repeat_eligibility=ExternalEffectRepeatEligibilityResult(
            verdict=ExternalEffectRepeatEligibilityVerdict.ELIGIBLE,
            reason=ExternalEffectRepeatEligibilityReason.ALLOWED_IDEMPOTENT_REPEAT,
            invocation_id=inv.invocation_id,
            idempotency_key=inv.idempotency_key,
        ),
    )
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_escalate_required_posture_forged_repeat_blocked() -> None:
    contract = ExternalEffectContract(
        contract_id="test.escalate.r1",
        operation_key="external_work.create_work",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
    )
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=contract,
        repeat_eligibility=ExternalEffectRepeatEligibilityResult(
            verdict=ExternalEffectRepeatEligibilityVerdict.ELIGIBLE,
            reason=ExternalEffectRepeatEligibilityReason.ALLOWED_IDEMPOTENT_REPEAT,
            invocation_id=inv.invocation_id,
            idempotency_key=inv.idempotency_key,
        ),
    )
    decision = _repeat_decision_for(inv, inv.idempotency_key)
    _execute_repeat_blocked(request, decision)


def test_production_hitl_uses_governed_continuation_store() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    inv = _invocation().model_copy(
        update={"task_id": str(task_id), "run_id": str(run_id)},
    )
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=None,
        dispatch_state=ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY,
        effect_contract=_CREATE_CONTRACT,
    )
    from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
        evaluate_provider_invocation_recovery,
    )

    decision = evaluate_provider_invocation_recovery(request)
    store = InMemoryContinuationStateStore()
    hitl_port = GovernedExternalWorkProviderRecoveryHitlPort(
        continuation_store=store,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(hitl=hitl_port),
    )
    assert result.provider_mutation_count == 0
    stored = store.get_continuation(inv.task_id)
    assert stored is not None
    assert result.hitl is not None
    assert stored.continuation_request_id == result.hitl.governed_continuation_request_id


def test_governed_repeat_lifecycle_fresh_authorization_and_durable_ordering() -> None:
    fake = DeterministicExternalWorkFake(
        fail_create_with_code={_FAIL_IDEMP: ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE},
    )
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime, _ = _build_runtime(fake, task_id)
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        step, _ = _create_step(
            runtime,
            fake,
            task_id,
            run_id,
            attempt_id,
            execution_id,
            idempotency_key=_FAIL_IDEMP,
        )
    assert fake.create_calls == 1
    assert step.governed_result is None
    inv_id = str(step.adapter_result.metadata.get("provider_invocation_id", ""))
    original_inv = runtime.orchestrator._provider_invocation_store.get_invocation(inv_id)
    assert original_inv is not None
    original_outcome = runtime.orchestrator._provider_invocation_store.get_outcome(
        original_inv.invocation_id,
    )
    assert original_outcome is not None
    assert original_outcome.status is ProviderInvocationStatus.FAILED

    fake._fail_create_with_code.clear()

    def _meta(inv: object) -> dict[str, object]:
        from intergrax.contracts.provider_invocation import ProviderInvocation

        assert isinstance(inv, ProviderInvocation)
        return _create_meta(inv.task_id, inv.run_id, idempotency_key=_FAIL_IDEMP)

    repeat_port = GovernedExternalWorkProviderRecoveryRepeatPort(
        orchestrator=runtime.orchestrator,
        adapter=runtime.adapter,
        invocation_store=runtime.orchestrator._provider_invocation_store,
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        metadata_for_invocation=_meta,
        clock=lambda: _T0,
        host_execution_id="exec-gr7a7r1-repeat",
    )

    recovery = GovernedExternalWorkProviderRecovery.build(
        GovernedExternalWorkProviderReconciliation.build(fake),
    )
    request = recovery.build_recovery_request(
        invocation=original_inv,
        outcome=original_outcome,
        capabilities=quote_first_partner_capability_fixture(provider_id=original_inv.provider_id),
        repeat_policy=_AllowRepeatPolicy(),
    )
    decision = recovery.decide(request)
    assert decision.action is ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT

    eval_before = runtime.policy_evaluator.last_evaluation
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = execute_provider_invocation_recovery(
            request,
            decision=decision,
            ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
        )
    assert fake.create_calls == 2
    assert runtime.policy_evaluator.last_evaluation is not eval_before
    assert result.provider_mutation_count == 1
    assert result.repeat is not None
    assert result.repeat.repeat_invocation_id != original_inv.invocation_id
    assert result.repeat.idempotency_key == original_inv.idempotency_key

    repeat_inv = runtime.orchestrator._provider_invocation_store.get_invocation(
        result.repeat.repeat_invocation_id,
    )
    repeat_outcome = runtime.orchestrator._provider_invocation_store.get_outcome(
        result.repeat.repeat_invocation_id,
    )
    assert repeat_inv is not None
    assert repeat_outcome is not None
    assert repeat_inv.provider_id == original_inv.provider_id
    assert repeat_inv.operation == original_inv.operation
    unchanged = runtime.orchestrator._provider_invocation_store.get_invocation(
        original_inv.invocation_id,
    )
    assert unchanged == original_inv
