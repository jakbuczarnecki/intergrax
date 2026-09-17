# © Artur Czarnecki. All rights reserved.

"""GR-7-A5 — external-effect repeat eligibility foundation tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agents.external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from agents.external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.enterprise_reliability import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectRepeatEligibilityReason,
    ExternalEffectRepeatEligibilityRequest,
    ExternalEffectRepeatEligibilityVerdict,
    ExternalEffectRepeatPolicyDecision,
    ExternalEffectRepeatPolicyRequest,
    ExternalEffectSafetyCapabilities,
    UnknownUncertaintyPosture,
    evaluate_external_effect_repeat_eligibility,
    evaluate_unknown_uncertainty_posture,
)
from intergrax.contracts.external_work import ExternalWorkErrorCode
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.enterprise_reliability.default_repeat_eligibility_policy import (
    DefaultDenyExternalEffectRepeatPolicy,
    default_deny_external_effect_repeat_policy,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 3, 17, 12, 0, 0, tzinfo=UTC)


def _caps(**updates: object) -> ExternalWorkProviderCapabilities:
    base = ExternalWorkProviderCapabilities(
        provider_id="test-provider",
        supports_create=True,
        supports_accept=True,
        supports_cancel=True,
        supports_status_polling=True,
        supports_idempotency=True,
    )
    return base.model_copy(update=updates)


class _AllowRepeatPolicy:
    policy_id = "test_allow_repeat"

    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        _ = request
        return ExternalEffectRepeatPolicyDecision(allow_repeat=True)


class _DenyRepeatPolicy:
    policy_id = "test_deny_repeat"

    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        _ = request
        return ExternalEffectRepeatPolicyDecision(allow_repeat=False)


class _ExplodingPolicy:
    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        _ = request
        raise RuntimeError("policy broken")


@dataclass
class _RecordingPolicy:
    allow: bool = True
    calls: int = 0
    last_request: ExternalEffectRepeatPolicyRequest | None = field(default=None)

    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        self.calls += 1
        self.last_request = request
        return ExternalEffectRepeatPolicyDecision(allow_repeat=self.allow)


class _TransientOnlyRepeatPolicy:
    policy_id = "transient_only"

    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        code = request.outcome.error_code
        allow = code == ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE.value
        return ExternalEffectRepeatPolicyDecision(allow_repeat=allow)


class _DenyPermanentFailedPolicy:
    policy_id = "deny_permanent_failed"

    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        if request.outcome.status is not ProviderInvocationStatus.FAILED:
            return ExternalEffectRepeatPolicyDecision(allow_repeat=True)
        permanent = request.outcome.error_code == (
            ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE.value
        )
        return ExternalEffectRepeatPolicyDecision(allow_repeat=not permanent)


def _invocation(
    *,
    operation: str = "external_work.create_work",
    idempotency_key: str | None = "idem-1",
) -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id="inv-1",
        provider_id="provider-a",
        operation=operation,
        task_id="task-1",
        run_id="run-1",
        idempotency_key=idempotency_key,
        request_digest="digest",
        started_at=_NOW,
    )


def _outcome(
    status: ProviderInvocationStatus,
    *,
    error_code: str | None = None,
    external_status: str | None = None,
    provider_request_id: str | None = None,
    provider_operation_id: str | None = None,
    invocation_id: str = "inv-1",
) -> ProviderInvocationOutcome:
    return ProviderInvocationOutcome(
        invocation_id=invocation_id,
        status=status,
        completed_at=_NOW,
        error_code=error_code,
        external_status=external_status,
        provider_request_id=provider_request_id,
        provider_operation_id=provider_operation_id,
    )


def _contract(
    *,
    operation_key: str = "external_work.create_work",
    idempotency: ExternalEffectCapabilitySupport,
    reconciliation: ExternalEffectCapabilitySupport = ExternalEffectCapabilitySupport.NOT_SUPPORTED,
) -> ExternalEffectContract:
    probes = ("get_work",) if reconciliation is ExternalEffectCapabilitySupport.SUPPORTED else ()
    return ExternalEffectContract(
        contract_id="c-1",
        operation_key=operation_key,
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=idempotency,
            reconciliation=reconciliation,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=probes,
    )


def _evaluate(
    *,
    invocation: ProviderInvocation | None = _invocation(),
    outcome: ProviderInvocationOutcome | None = _outcome(ProviderInvocationStatus.UNKNOWN),
    contract: ExternalEffectContract | None = None,
    policy: object | None = _AllowRepeatPolicy(),
):
    if contract is None:
        contract = _contract(idempotency=ExternalEffectCapabilitySupport.SUPPORTED)
    return evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=invocation,
            outcome=outcome,
            effect_contract=contract,
        ),
        policy=policy,
    )


def test_succeeded_retry_not_allowed() -> None:
    result = _evaluate(outcome=_outcome(ProviderInvocationStatus.SUCCEEDED))
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_ALREADY_SUCCEEDED


def test_create_unknown_idempotency_supported_policy_allows_eligible() -> None:
    caps = _caps()
    contract = external_work_effect_contract_for_action(ACTION_CREATE_EXTERNAL_WORK, caps)
    assert evaluate_unknown_uncertainty_posture(contract) is (
        UnknownUncertaintyPosture.RECONCILE_OR_IDEMPOTENT_REPEAT
    )
    result = _evaluate(
        contract=contract,
        policy=_AllowRepeatPolicy(),
    )
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.ELIGIBLE
    assert result.reason is ExternalEffectRepeatEligibilityReason.ALLOWED_IDEMPOTENT_REPEAT
    assert result.idempotency_key == "idem-1"


def test_create_unknown_no_idempotency_not_allowed() -> None:
    caps = _caps(supports_idempotency=False)
    contract = external_work_effect_contract_for_action(ACTION_CREATE_EXTERNAL_WORK, caps)
    result = _evaluate(contract=contract, policy=_AllowRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_NOT_SUPPORTED


def test_create_unknown_idempotency_policy_denies() -> None:
    result = _evaluate(policy=_DenyRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_POLICY


def test_cancel_unknown_not_allowed_despite_provider_idempotency_flag() -> None:
    caps = _caps()
    contract = external_work_effect_contract_for_action(ACTION_CANCEL_EXTERNAL_WORK, caps)
    assert contract.safety.idempotency is ExternalEffectCapabilitySupport.NOT_SUPPORTED
    inv = _invocation(operation="external_work.cancel_work")
    result = _evaluate(
        invocation=inv,
        contract=contract,
        policy=_AllowRepeatPolicy(),
    )
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_NOT_SUPPORTED


def test_accept_unknown_may_be_eligible_when_contract_supports_idempotency() -> None:
    caps = _caps()
    contract = external_work_effect_contract_for_action(ACTION_ACCEPT_QUOTE, caps)
    inv = _invocation(operation="external_work.accept_quote")
    result = _evaluate(invocation=inv, contract=contract, policy=_AllowRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.ELIGIBLE


def test_reconcile_only_posture_denies_repeat() -> None:
    contract = _contract(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
    )
    assert evaluate_unknown_uncertainty_posture(contract) is UnknownUncertaintyPosture.RECONCILE_ONLY
    result = _evaluate(contract=contract, policy=_AllowRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason in {
        ExternalEffectRepeatEligibilityReason.DENIED_RECONCILIATION_REQUIRED,
        ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_NOT_SUPPORTED,
    }


def test_missing_idempotency_key_not_allowed() -> None:
    result = _evaluate(invocation=_invocation(idempotency_key=None), policy=_AllowRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_KEY_MISSING


def test_outcome_missing_not_allowed() -> None:
    result = _evaluate(outcome=None, policy=_AllowRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_OUTCOME_MISSING


def test_governance_deny_no_invocation_not_applicable() -> None:
    result = _evaluate(invocation=None, policy=_AllowRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_APPLICABLE
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_NO_PROVIDER_ATTEMPT


def test_default_deny_policy_without_explicit_allow() -> None:
    result = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=_invocation(),
            outcome=_outcome(ProviderInvocationStatus.UNKNOWN),
            effect_contract=_contract(idempotency=ExternalEffectCapabilitySupport.SUPPORTED),
        ),
        policy=default_deny_external_effect_repeat_policy(),
    )
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_POLICY
    assert result.policy_id == DefaultDenyExternalEffectRepeatPolicy().policy_id


def test_no_policy_configured_fail_closed() -> None:
    result = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=_invocation(),
            outcome=_outcome(ProviderInvocationStatus.UNKNOWN),
            effect_contract=_contract(idempotency=ExternalEffectCapabilitySupport.SUPPORTED),
        ),
        policy=None,
    )
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_POLICY


def test_policy_pluginability_custom_allow_deny() -> None:
    eligible = _evaluate(policy=_AllowRepeatPolicy())
    denied = _evaluate(policy=_DenyRepeatPolicy())
    assert eligible.verdict is ExternalEffectRepeatEligibilityVerdict.ELIGIBLE
    assert denied.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED


def test_policy_evaluation_failure_fail_closed() -> None:
    result = _evaluate(policy=_ExplodingPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_POLICY


def test_eligible_binds_original_idempotency_key() -> None:
    result = _evaluate(invocation=_invocation(idempotency_key="stable-key-42"))
    assert result.idempotency_key == "stable-key-42"


def test_policy_receives_full_provider_invocation_outcome() -> None:
    outcome = _outcome(
        ProviderInvocationStatus.UNKNOWN,
        error_code="ignored-for-unknown",
        external_status="ext-pending",
        provider_request_id="req-99",
        provider_operation_id="op-88",
    )
    recorder = _RecordingPolicy()
    _evaluate(outcome=outcome, policy=recorder)
    assert recorder.calls == 1
    assert recorder.last_request is not None
    req = recorder.last_request
    assert isinstance(req.outcome, ProviderInvocationOutcome)
    assert req.outcome.status is ProviderInvocationStatus.UNKNOWN
    assert req.outcome.error_code == "ignored-for-unknown"
    assert req.outcome.external_status == "ext-pending"
    assert req.outcome.provider_request_id == "req-99"
    assert req.outcome.provider_operation_id == "op-88"
    assert not hasattr(req, "outcome_status")


def test_failed_not_auto_eligible_without_policy_allow() -> None:
    result = _evaluate(
        outcome=_outcome(ProviderInvocationStatus.FAILED),
        policy=default_deny_external_effect_repeat_policy(),
    )
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_POLICY


def test_failed_transient_error_policy_allows_eligible() -> None:
    outcome = _outcome(
        ProviderInvocationStatus.FAILED,
        error_code=ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE.value,
        external_status="503",
    )
    result = _evaluate(outcome=outcome, policy=_TransientOnlyRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.ELIGIBLE


def test_failed_permanent_error_policy_denies() -> None:
    outcome = _outcome(
        ProviderInvocationStatus.FAILED,
        error_code=ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE.value,
    )
    result = _evaluate(outcome=outcome, policy=_DenyPermanentFailedPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_POLICY


def test_failed_policy_inspects_error_code_not_core_hardcode() -> None:
    recorder = _RecordingPolicy(allow=False)
    outcome = _outcome(
        ProviderInvocationStatus.FAILED,
        error_code=ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE.value,
    )
    _evaluate(outcome=outcome, policy=recorder)
    assert recorder.last_request is not None
    assert (
        recorder.last_request.outcome.error_code
        == ExternalWorkErrorCode.TRANSIENT_REMOTE_FAILURE.value
    )


def test_policy_not_called_for_succeeded() -> None:
    recorder = _RecordingPolicy()
    _evaluate(outcome=_outcome(ProviderInvocationStatus.SUCCEEDED), policy=recorder)
    assert recorder.calls == 0


def test_policy_not_called_without_idempotency_support() -> None:
    caps = _caps(supports_idempotency=False)
    contract = external_work_effect_contract_for_action(ACTION_CREATE_EXTERNAL_WORK, caps)
    recorder = _RecordingPolicy()
    _evaluate(contract=contract, policy=recorder)
    assert recorder.calls == 0


def test_policy_not_called_for_reconcile_only_posture() -> None:
    contract = _contract(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
    )
    recorder = _RecordingPolicy()
    _evaluate(contract=contract, policy=recorder)
    assert recorder.calls == 0


def test_policy_not_called_for_escalate_required_posture() -> None:
    contract = _contract(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
    )
    assert evaluate_unknown_uncertainty_posture(contract) is (
        UnknownUncertaintyPosture.ESCALATE_REQUIRED
    )
    recorder = _RecordingPolicy()
    result = _evaluate(contract=contract, policy=recorder)
    assert recorder.calls == 0
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_NOT_SUPPORTED


def test_create_unknown_reconcile_only_hard_denied() -> None:
    contract = _contract(
        idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
    )
    result = _evaluate(contract=contract, policy=_AllowRepeatPolicy())
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED


def test_outcome_invocation_mismatch_policy_not_called() -> None:
    recorder = _RecordingPolicy()
    result = _evaluate(
        outcome=_outcome(ProviderInvocationStatus.UNKNOWN, invocation_id="other"),
        policy=recorder,
    )
    assert recorder.calls == 0
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    assert result.reason is ExternalEffectRepeatEligibilityReason.DENIED_INVALID_STATE


def test_operation_mismatch_policy_not_called() -> None:
    recorder = _RecordingPolicy()
    result = _evaluate(
        invocation=_invocation(operation="external_work.cancel_work"),
        policy=recorder,
    )
    assert recorder.calls == 0
    assert result.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED


def test_repeat_eligibility_module_has_no_external_work_integration_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[3]
        / "intergrax"
        / "contracts"
        / "enterprise_reliability"
        / "repeat_eligibility.py"
    )
    source = module_path.read_text(encoding="utf-8")
    assert "ExternalWorkProviderCapabilities" not in source
    assert "ExternalWorkAdapterResult" not in source
    assert "ExternalWorkIntegration" not in source
