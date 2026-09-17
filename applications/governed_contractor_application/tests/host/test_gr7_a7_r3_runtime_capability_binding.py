# © Artur Czarnecki. All rights reserved.

"""GR-7-A7-R3 — execution-time repeat port capability binding."""

from __future__ import annotations

from dataclasses import dataclass
import pytest

from applications.governed_contractor_application.tests.host.test_gr7_a7_provider_recovery import (
    _AllowRepeatPolicy,
    _RecordingRepeatPort,
    _SelectIdempotentRepeatRecoveryPolicy,
    _invocation,
    _outcome,
    _repeat_eligibility,
)
from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDecision,
    ProviderInvocationRecoveryReason,
    ProviderInvocationRecoveryRequest,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityRequest,
    evaluate_external_effect_repeat_eligibility,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.governed_execution_result import (
    external_work_decision_action_for_provider_operation,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryExecutionBlockReason,
    ProviderInvocationRecoveryExecutionDisposition,
    ProviderInvocationRecoveryExecutionPorts,
    ProviderInvocationRecoveryRepeatResult,
    decide_provider_invocation_recovery,
    execute_provider_invocation_recovery,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CAPABILITIES = quote_first_partner_capability_fixture()
_CREATE_CONTRACT = external_work_effect_contract_for_action(
    ACTION_CREATE_EXTERNAL_WORK,
    _CAPABILITIES,
)
_ACCEPT_CONTRACT = external_work_effect_contract_for_action(
    ACTION_ACCEPT_QUOTE,
    _CAPABILITIES,
)
_CANCEL_CONTRACT = external_work_effect_contract_for_action(
    ACTION_CANCEL_EXTERNAL_WORK,
    _CAPABILITIES,
)


def _repeat_decision(inv: ProviderInvocation) -> ProviderInvocationRecoveryDecision:
    return ProviderInvocationRecoveryDecision(
        action=ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,
        reason=ProviderInvocationRecoveryReason.POLICY_SELECTED,
        invocation_id=inv.invocation_id,
        idempotency_key=inv.idempotency_key,
    )


def _eligible_create_request(
    *,
    repeat_execution_supported: bool = True,
) -> ProviderInvocationRecoveryRequest:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    return ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
        repeat_execution_supported=repeat_execution_supported,
    )


@dataclass
class _ConfigurableRepeatPort:
    supports: bool
    calls: int = 0
    capability_raises: bool = False

    def supports_provider_operation(self, operation: str) -> bool:
        if self.capability_raises:
            raise RuntimeError("capability probe failed")
        if not self.supports:
            return False
        return (
            external_work_decision_action_for_provider_operation(operation)
            == ACTION_CREATE_EXTERNAL_WORK
        )

    def execute_idempotent_repeat(
        self,
        *,
        original_invocation: ProviderInvocation,
        original_outcome: ProviderInvocationOutcome,
        effect_contract: object,
    ) -> ProviderInvocationRecoveryRepeatResult:
        _ = original_outcome, effect_contract
        self.calls += 1
        return ProviderInvocationRecoveryRepeatResult(
            repeat_invocation_id=f"{original_invocation.invocation_id}-repeat",
            idempotency_key=original_invocation.idempotency_key or "",
            provider_mutation_count=1,
        )


def _assert_no_repeat_execution(
    result: object,
    repeat_port: _ConfigurableRepeatPort | _RecordingRepeatPort,
) -> None:
    from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
        ProviderInvocationRecoveryExecutionResult,
    )

    assert isinstance(result, ProviderInvocationRecoveryExecutionResult)
    assert repeat_port.calls == 0
    assert result.provider_mutation_count == 0
    assert result.disposition is ProviderInvocationRecoveryExecutionDisposition.BLOCKED


def _assert_repeat_blocked(
    result: object,
    repeat_port: _ConfigurableRepeatPort | _RecordingRepeatPort,
) -> None:
    from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
        ProviderInvocationRecoveryExecutionResult,
    )

    assert isinstance(result, ProviderInvocationRecoveryExecutionResult)
    assert repeat_port.calls == 0
    assert result.provider_mutation_count == 0
    assert result.disposition is ProviderInvocationRecoveryExecutionDisposition.BLOCKED
    assert (
        result.block_reason
        is ProviderInvocationRecoveryExecutionBlockReason.REPEAT_EXECUTION_UNSUPPORTED
    )


def test_snapshot_true_port_false_blocked() -> None:
    request = _eligible_create_request(repeat_execution_supported=True)
    inv = request.invocation
    assert inv is not None
    port = _ConfigurableRepeatPort(supports=False)
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    _assert_repeat_blocked(result, port)


def test_port_swapped_after_request_creation_blocked() -> None:
    port_at_decision = _ConfigurableRepeatPort(supports=True)
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
        repeat_execution_supported=port_at_decision.supports_provider_operation(
            inv.operation
        ),
    )
    assert request.repeat_execution_supported is True
    port_at_execution = _ConfigurableRepeatPort(supports=False)
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port_at_execution),
    )
    _assert_repeat_blocked(result, port_at_execution)


def test_forged_snapshot_true_port_false_blocked() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
        repeat_execution_supported=True,
    )
    port = _ConfigurableRepeatPort(supports=False)
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    _assert_repeat_blocked(result, port)


def test_snapshot_false_port_true_blocked() -> None:
    request = _eligible_create_request(repeat_execution_supported=False)
    inv = request.invocation
    assert inv is not None
    port = _ConfigurableRepeatPort(supports=True)
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    _assert_repeat_blocked(result, port)


def test_capability_method_exception_fail_closed() -> None:
    request = _eligible_create_request(repeat_execution_supported=True)
    inv = request.invocation
    assert inv is not None
    port = _ConfigurableRepeatPort(supports=True, capability_raises=True)
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    _assert_repeat_blocked(result, port)


def test_happy_path_snapshot_and_port_supported_one_mutation() -> None:
    request = _eligible_create_request(repeat_execution_supported=True)
    inv = request.invocation
    assert inv is not None
    port = _RecordingRepeatPort(supported=True)
    result = execute_provider_invocation_recovery(
        request,
        decision=decide_provider_invocation_recovery(
            request,
            policy=_SelectIdempotentRepeatRecoveryPolicy(),
        ),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    assert port.calls == 1
    assert result.provider_mutation_count == 1
    assert result.disposition is ProviderInvocationRecoveryExecutionDisposition.COMPLETED
    assert result.repeat is not None
    assert result.repeat.repeat_invocation_id != inv.invocation_id
    assert result.repeat.idempotency_key == inv.idempotency_key


def test_custom_port_pluginability_when_both_gates_true() -> None:
    @dataclass
    class _CustomCreateRepeatPort:
        calls: int = 0

        def supports_provider_operation(self, operation: str) -> bool:
            return operation == "external_work.create_work"

        def execute_idempotent_repeat(
            self,
            *,
            original_invocation: ProviderInvocation,
            original_outcome: ProviderInvocationOutcome,
            effect_contract: object,
        ) -> ProviderInvocationRecoveryRepeatResult:
            _ = original_outcome, effect_contract
            self.calls += 1
            return ProviderInvocationRecoveryRepeatResult(
                repeat_invocation_id=f"{original_invocation.invocation_id}-custom",
                idempotency_key=original_invocation.idempotency_key or "",
                provider_mutation_count=1,
            )

    request = _eligible_create_request(repeat_execution_supported=True)
    inv = request.invocation
    assert inv is not None
    port = _CustomCreateRepeatPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    assert port.calls == 1
    assert result.provider_mutation_count == 1


def test_unknown_operation_no_execution() -> None:
    inv = _invocation().model_copy(update={"operation": "unknown.provider.op"})
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
        repeat_execution_supported=True,
    )
    port = _ConfigurableRepeatPort(supports=True)
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    _assert_no_repeat_execution(result, port)


def _accept_repeat_request() -> ProviderInvocationRecoveryRequest:
    inv = _invocation().model_copy(update={"operation": "submit_quote_acceptance"})
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    eligibility = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=inv,
            outcome=outcome,
            effect_contract=_ACCEPT_CONTRACT,
        ),
        policy=_AllowRepeatPolicy(),
    )
    return ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_ACCEPT_CONTRACT,
        repeat_eligibility=eligibility,
        repeat_execution_supported=True,
    )


def test_accept_unsupported_no_repeat_execute() -> None:
    request = _accept_repeat_request()
    inv = request.invocation
    assert inv is not None
    port = _ConfigurableRepeatPort(supports=True)
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    _assert_repeat_blocked(result, port)


def test_cancel_unsupported_no_repeat_execute() -> None:
    inv = _invocation().model_copy(update={"operation": "external_work.cancel_work"})
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    eligibility = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=inv,
            outcome=outcome,
            effect_contract=_CANCEL_CONTRACT,
        ),
        policy=_AllowRepeatPolicy(),
    )
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CANCEL_CONTRACT,
        repeat_eligibility=eligibility,
        repeat_execution_supported=True,
    )
    port = _ConfigurableRepeatPort(supports=True)
    result = execute_provider_invocation_recovery(
        request,
        decision=_repeat_decision(inv),
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=port),
    )
    _assert_no_repeat_execution(result, port)
