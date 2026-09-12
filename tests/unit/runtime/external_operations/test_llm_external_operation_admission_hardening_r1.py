# © Artur Czarnecki. All rights reserved.

"""LLM-EXTERNAL-OPERATION-ADMISSION R1 — runtime qualification matrix."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmissionContext,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.attempt import (
    ExternalOperationAttemptLifecycle,
)
from intergrax.contracts.external_operations.evidence import ProviderExecutionOutcome
from intergrax.contracts.external_operations.intent import (
    ExternalOperationIntent,
    ExternalOperationType,
    mint_external_operation_intent_id,
)
from intergrax.contracts.external_operations.provider import (
    ProviderPayloadBounds,
    ProviderRiskProfile,
)
from intergrax.contracts.external_operations.safety import (
    ExternalOperationAdmissionDeniedError,
    ExternalOperationApprovalRequiredError,
    assert_no_secrets_in_audit_payload,
)
from intergrax.contracts.external_operations.governance import (
    ExternalOperationGovernanceContext,
)
from intergrax.runtime.external_operations.admission.audit_chain import (
    InMemoryExternalOperationAuditChain,
)
from intergrax.runtime.external_operations.admission.execution_gate import (
    ExternalOperationExecutionGate,
)
from intergrax.runtime.external_operations.admission.governance_bridge import (
    resolve_governance_admission,
)
from intergrax.runtime.external_operations.admission.local_admission import (
    PolicyExternalOperationAdmission,
)
from intergrax.runtime.external_operations.admission.provider_executor import (
    ContainedProviderExecutionError,
    execute_contained_provider_call,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _intent(
    *,
    tenant_id: str = "tenant_a",
    target: str = "dev:connector",
    operation_type: ExternalOperationType = ExternalOperationType.INTEGRATION_INVOKE,
) -> ExternalOperationIntent:
    return ExternalOperationIntent(
        intent_id=mint_external_operation_intent_id(),
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        operation_type=operation_type,
        target_resource=target,
        requested_by="operator",
        justification="restart connector after diagnostic",
        created_at=datetime.now(timezone.utc),
    )


class _StubProvider:
    def __init__(self, *, provider_id: str, fail: bool = False) -> None:
        self._provider_id = provider_id
        self._fail = fail
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def version(self) -> str:
        return "1"

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"invoke"})

    @property
    def tenant_scope(self) -> frozenset[str] | None:
        return None

    @property
    def risk_profile(self) -> ProviderRiskProfile:
        return ProviderRiskProfile.LOW

    @property
    def payload_bounds(self) -> ProviderPayloadBounds:
        return ProviderPayloadBounds(
            max_payload_bytes=1024,
            timeout_seconds=5.0,
            max_retries=0,
        )

    def execute_admitted(self, attempt: object) -> ProviderExecutionOutcome:
        self.calls += 1
        if self._fail:
            raise RuntimeError("provider unavailable")
        return ProviderExecutionOutcome(status="success", safe_summary="ok")


def test_denied_operation_never_reaches_provider() -> None:
    admission = PolicyExternalOperationAdmission()
    gate = ExternalOperationExecutionGate(admission=admission)
    provider = _StubProvider(provider_id="sap")
    intent = _intent(
        target="production:connector",
        operation_type=ExternalOperationType.CONNECTOR_RESTART,
    )
    with pytest.raises(ExternalOperationApprovalRequiredError):
        gate.admit_intent(
            intent,
            context=ExternalOperationAdmissionContext(
                tenant_id="tenant_a",
                production_target=True,
            ),
            provider_id=provider.provider_id,
        )
    assert provider.calls == 0


def test_human_approval_required_blocks_execution() -> None:
    admission = PolicyExternalOperationAdmission()
    gate = ExternalOperationExecutionGate(admission=admission)
    intent = _intent(
        target="production:crm",
        operation_type=ExternalOperationType.CONNECTOR_RESTART,
    )
    with pytest.raises(ExternalOperationApprovalRequiredError, match="production"):
        gate.admit_intent(
            intent,
            context=ExternalOperationAdmissionContext(
                tenant_id="tenant_a",
                production_target=True,
            ),
        )


def test_governance_denied_never_executes() -> None:
    admission = PolicyExternalOperationAdmission()
    intent = _intent()
    result = resolve_governance_admission(
        intent=intent,
        governance=ExternalOperationGovernanceContext(
            decision_id="decision_" + "c" * 32,
            governance_approved=False,
            tenant_id="tenant_a",
        ),
        admission=admission,
    )
    assert result.may_execute is False
    assert result.admission.verdict is OperationAdmissionVerdict.DENY


def test_provider_failure_is_contained() -> None:
    admission = PolicyExternalOperationAdmission()
    gate = ExternalOperationExecutionGate(admission=admission)
    provider = _StubProvider(provider_id="slack", fail=True)
    intent = _intent(target="dev:slack")
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id="tenant_a"),
        provider_id=provider.provider_id,
    )
    with pytest.raises(ContainedProviderExecutionError) as exc_info:
        execute_contained_provider_call(
            gate=gate,
            provider=provider,
            attempt=admitted,
            fn=lambda: provider.execute_admitted(admitted),
        )
    assert exc_info.value.evidence.kind.value == "PROVIDER_FAILURE"


def test_operation_attempt_is_reconstructable() -> None:
    audit = InMemoryExternalOperationAuditChain()
    admission = PolicyExternalOperationAdmission()
    gate = ExternalOperationExecutionGate(admission=admission, audit_chain=audit)
    intent = _intent()
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id="tenant_a"),
        provider_id="crm",
    )
    executing = gate.begin_execution(admitted.bind_provider("crm"))
    gate.complete_success(executing)
    chain = audit.reconstruct(admitted.operation_attempt_id)
    assert len(chain) >= 2
    assert chain[0].admission_decision.verdict is OperationAdmissionVerdict.ALLOW
    assert chain[-1].execution_status is ExternalOperationAttemptLifecycle.SUCCEEDED


def test_failed_external_operation_creates_evidence() -> None:
    admission = PolicyExternalOperationAdmission()
    gate = ExternalOperationExecutionGate(admission=admission)
    provider = _StubProvider(provider_id="cloud", fail=True)
    intent = _intent(target="dev:cloud")
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id="tenant_a"),
        provider_id=provider.provider_id,
    )
    with pytest.raises(ContainedProviderExecutionError) as exc_info:
        execute_contained_provider_call(
            gate=gate,
            provider=provider,
            attempt=admitted,
            fn=lambda: (_ for _ in ()).throw(RuntimeError("timeout")),
        )
    assert "provider" in exc_info.value.evidence.safe_summary.lower()


def test_operation_attempt_cannot_cross_tenant() -> None:
    admission = PolicyExternalOperationAdmission()
    gate = ExternalOperationExecutionGate(admission=admission)
    intent = _intent(tenant_id="tenant_a")
    with pytest.raises(ExternalOperationAdmissionDeniedError):
        gate.admit_intent(
            intent,
            context=ExternalOperationAdmissionContext(tenant_id="tenant_b"),
        )


def test_secret_not_present_in_audit() -> None:
    with pytest.raises(ValueError, match="secret"):
        assert_no_secrets_in_audit_payload(
            ("Authorization: Bearer sk-" + "x" * 40,),
        )
