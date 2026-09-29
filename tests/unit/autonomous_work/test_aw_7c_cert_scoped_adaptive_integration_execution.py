# © Artur Czarnecki. All rights reserved.

"""AW-7C-CERT — adversarial scoped adaptive integration execution qualification."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from intergrax.autonomous_work.scoped_adaptive_integration_execution import (
    WorkerScopedAdaptiveIntegrationExecutionCoordinator,
)
from intergrax.capability_qualification.qualification_service import CapabilityQualificationService
from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionOutcome,
    validate_execution_bound_qualification_proof,
)
from intergrax.contracts.execution_identity import ExecutionId, mint_execution_id
from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
)
from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialScopeAdmissionDecision,
    CredentialUseGrant,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
from intergrax.integrations.credentials.secrets_store_resolver import SecretsStoreCredentialResolver
from intergrax.integrations.qualification.reference_scoped_adaptive_integration_execution import (
    ReferenceScopedAdaptedIntegrationOperation,
    execute_reference_scoped_adaptive_integration,
)
from intergrax.integrations.qualification.reference_scoped_integration_adaptation import (
    REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
    reference_read_operation,
    reference_write_operation,
)
from intergrax.integrations.qualification.reference_scoped_integration_qualification import (
    ReferenceScopedIntegrationQualificationProvider,
)
from intergrax.integrations.qualification.scoped_adaptive_integration_execution_intake import (
    ScopedAdaptiveIntegrationReferenceExecutionIntake,
)
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities
from tests.unit.autonomous_work.test_aw_7c_p4_scoped_adaptive_integration_execution import (
    _TENANT,
    _TS,
    _dispatch_stack,
    _p4_request,
    _prepare,
)

pytestmark = pytest.mark.unit

_HOST_A = NetworkEgressHost(scheme="https", hostname="a.example.com", port=443)
_ALLOWLIST = NetworkEgressAllowlist(hosts=(_HOST_A,))


class _SecretsStore:
    def get_secret(self, path: str, *, version: str | None = None) -> str:
        return "opaque-token"


class _AllowAdmission:
    def admit(self, grant: CredentialUseGrant, scope: object) -> CredentialScopeAdmissionDecision:
        return CredentialScopeAdmissionDecision.ALLOW


def _sandbox_ok() -> SandboxSecurityCapabilities:
    return SandboxSecurityCapabilities(
        isolation_tier="local",
        provider_id="test",
        network_egress_allowlist_enforced=True,
        enforced_network_hosts=_ALLOWLIST,
    )


def _grant_for_execution(execution_id: ExecutionId, *, operation: str = "READ_CONFIGURATION") -> CredentialUseGrant:
    return CredentialUseGrant(
        grant_id="grant-1",
        credential_ref=CredentialRef.from_secret_path(
            provider_id=REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
            secret_path="secrets/tenant-a/demo",
            tenant_id=_TENANT,
        ),
        tenant_id=_TENANT,
        provider_id=REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
        integration_id="rs-1",
        operation=operation,
        execution_id=str(execution_id),
        target_scope=_ALLOWLIST,
        expires_at=_TS + timedelta(hours=1),
    )


def _broker() -> ScopedCredentialBroker:
    return ScopedCredentialBroker(
        resolver=SecretsStoreCredentialResolver(_SecretsStore()),
        admission=_AllowAdmission(),
        time_provider=type(
            "_Fixed",
            (),
            {"utc_now": staticmethod(lambda: _TS)},
        ),
    )


def _cert_intake(
    *,
    operation: str = "READ_CONFIGURATION",
    sandbox: SandboxSecurityCapabilities | None = None,
) -> ScopedAdaptiveIntegrationReferenceExecutionIntake:
    broker = _broker()
    return ScopedAdaptiveIntegrationReferenceExecutionIntake(
        credential_broker=broker,
        credential_grant_for_execution=lambda eid: _grant_for_execution(eid, operation=operation),
        sandbox_capabilities=sandbox or _sandbox_ok(),
        operation_port=ReferenceScopedAdaptedIntegrationOperation(),
    )


@pytest.mark.asyncio
async def test_cert_full_reference_e2e() -> None:
    preparation = _prepare()
    intake = _cert_intake()
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert result.execution_id is not None
    op = intake.operation_port
    assert type(op) is ReferenceScopedAdaptedIntegrationOperation
    assert op.last_operation == "READ_CONFIGURATION"


@pytest.mark.asyncio
async def test_cert_requested_operation_not_permitted_rejected() -> None:
    preparation = _prepare()
    intake = _cert_intake()
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    bad = replace(_p4_request(preparation), requested_operation=reference_write_operation())
    bad = replace(bad, execution_idempotency_key="idem-bad-op")
    result = await coordinator.execute(bad)
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED


def test_cert_forged_handoff_missing_qualification_proof() -> None:
    preparation = _prepare()
    qual_request = preparation.qualification_request
    assert qual_request is not None
    decision = CapabilityQualificationService(
        (ReferenceScopedIntegrationQualificationProvider(),),
    ).qualify(qual_request)
    from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
        build_scoped_adaptive_integration_execution_handoff,
    )

    handoff = build_scoped_adaptive_integration_execution_handoff(
        preparation=preparation,
        qualification_request=qual_request,
        accepted_qualification=decision,
        requested_operation=reference_read_operation(),
        execution_idempotency_key="forged",
    )
    tampered = replace(handoff, qualification_request_id="forged-request-id")
    assert validate_execution_bound_qualification_proof(tampered) is not None
    execution_id = mint_execution_id()
    operation = ReferenceScopedAdaptedIntegrationOperation()
    operation.expected_operation = "READ_CONFIGURATION"
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=tampered,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_capabilities=_sandbox_ok(),
        credential_broker=_broker(),
        credential_grant=_grant_for_execution(execution_id),
        operation_port=operation,
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED
    assert operation.last_operation is None


@pytest.mark.parametrize(
    ("grant_id", "expected"),
    [
        ("grant-other", ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED),
        ("grant-1", ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED),
    ],
)
def test_cert_credential_grant_id_binding(
    grant_id: str,
    expected: ScopedAdaptiveIntegrationExecutionOutcome,
) -> None:
    preparation = _prepare()
    qual_request = preparation.qualification_request
    assert qual_request is not None
    decision = CapabilityQualificationService(
        (ReferenceScopedIntegrationQualificationProvider(),),
    ).qualify(qual_request)
    from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
        build_scoped_adaptive_integration_execution_handoff,
    )

    handoff = build_scoped_adaptive_integration_execution_handoff(
        preparation=preparation,
        qualification_request=qual_request,
        accepted_qualification=decision,
        requested_operation=reference_read_operation(),
        execution_idempotency_key="idem-grant",
    )
    execution_id = mint_execution_id()
    grant = _grant_for_execution(execution_id)
    grant = replace(grant, grant_id=grant_id)
    operation = ReferenceScopedAdaptedIntegrationOperation()
    operation.expected_operation = "READ_CONFIGURATION"
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_capabilities=_sandbox_ok(),
        credential_broker=_broker(),
        credential_grant=grant,
        operation_port=operation,
    )
    assert envelope.outcome is expected
    if expected is not ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED:
        assert operation.last_operation is None


@pytest.mark.parametrize(
    "sandbox",
    [
        SandboxSecurityCapabilities(
            isolation_tier="local",
            provider_id="test",
            network_egress_allowlist_enforced=False,
            enforced_network_hosts=_ALLOWLIST,
        ),
        SandboxSecurityCapabilities(
            isolation_tier="local",
            provider_id="test",
            network_egress_allowlist_enforced=True,
            enforced_network_hosts=None,
        ),
    ],
)
def test_cert_sandbox_attestation_matrix(sandbox: SandboxSecurityCapabilities) -> None:
    preparation = _prepare()
    qual_request = preparation.qualification_request
    assert qual_request is not None
    decision = CapabilityQualificationService(
        (ReferenceScopedIntegrationQualificationProvider(),),
    ).qualify(qual_request)
    from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
        build_scoped_adaptive_integration_execution_handoff,
    )

    handoff = build_scoped_adaptive_integration_execution_handoff(
        preparation=preparation,
        qualification_request=qual_request,
        accepted_qualification=decision,
        requested_operation=reference_read_operation(),
        execution_idempotency_key="idem-sbx",
    )
    execution_id = mint_execution_id()
    operation = ReferenceScopedAdaptedIntegrationOperation()
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_capabilities=sandbox,
        credential_broker=_broker(),
        credential_grant=_grant_for_execution(execution_id),
        operation_port=operation,
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.SANDBOX_SECURITY_UNSATISFIED
    assert operation.last_operation is None
