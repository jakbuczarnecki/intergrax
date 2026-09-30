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
    ReferenceExecutionBoundCredentialGrantProvider,
    ReferenceScopedAdaptedIntegrationEffectExecutor,
    ReferenceScopedAdaptedIntegrationEffectRequestPreparer,
    ReferenceScopedAdaptiveIntegrationSandboxSession,
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
    build_scoped_adaptive_integration_canonical_execution_intake,
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


def _sandbox_ok() -> ReferenceScopedAdaptiveIntegrationSandboxSession:
    return ReferenceScopedAdaptiveIntegrationSandboxSession(
        capabilities=SandboxSecurityCapabilities(
            isolation_tier="local",
            provider_id="test",
            network_egress_allowlist_enforced=True,
            enforced_network_hosts=_ALLOWLIST,
        ),
    )


def _grant_provider(
    *,
    grant_id: str = "grant-1",
    operation: str = "READ_CONFIGURATION",
) -> ReferenceExecutionBoundCredentialGrantProvider:
    return ReferenceExecutionBoundCredentialGrantProvider(
        grant_id=grant_id,
        credential_ref=CredentialRef.from_secret_path(
            provider_id=REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
            secret_path="secrets/tenant-a/demo",
            tenant_id=_TENANT,
        ),
        tenant_id=_TENANT,
        provider_id=REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
        integration_id="rs-1",
        target_scope=_ALLOWLIST,
        expires_at=_TS + timedelta(hours=1),
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


class _MismatchGrantProvider:
    def resolve_grant(
        self,
        *,
        execution_id: ExecutionId,
        credential_grant_ref: str,
        tenant_id: str,
        provider_id: str,
        integration_id: str,
        requested_operation: object,
    ) -> CredentialUseGrant:
        grant = _grant_for_execution(execution_id)
        return replace(grant, grant_id="grant-other")


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
    sandbox: ReferenceScopedAdaptiveIntegrationSandboxSession | None = None,
    grant_provider: ReferenceExecutionBoundCredentialGrantProvider | None = None,
) -> tuple[object, ReferenceScopedAdaptedIntegrationEffectRequestPreparer]:
    broker = _broker()
    intake, delegate = build_scoped_adaptive_integration_canonical_execution_intake(
        credential_broker=broker,
        credential_grant_provider=grant_provider or _grant_provider(),
        sandbox_security_source=sandbox or _sandbox_ok(),
    )
    preparer = delegate.effect_preparer
    assert type(preparer) is ReferenceScopedAdaptedIntegrationEffectRequestPreparer
    return intake, preparer


@pytest.mark.asyncio
async def test_cert_full_reference_e2e() -> None:
    preparation = _prepare()
    intake, preparer = _cert_intake()
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
    assert preparer.last_requested_operation == reference_read_operation()


@pytest.mark.asyncio
async def test_cert_requested_operation_not_permitted_rejected() -> None:
    preparation = _prepare()
    intake, _preparer = _cert_intake()
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
    preparer = ReferenceScopedAdaptedIntegrationEffectRequestPreparer()
    executor = ReferenceScopedAdaptedIntegrationEffectExecutor()
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=tampered,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_security_source=_sandbox_ok(),
        credential_broker=_broker(),
        credential_grant_provider=_grant_provider(),
        effect_preparer=preparer,
        effect_executor=executor,
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.QUALIFICATION_REJECTED
    assert preparer.last_requested_operation is None
    assert executor.call_count == 0


@pytest.mark.parametrize(
    ("use_mismatch_provider", "expected"),
    [
        (True, ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED),
        (False, ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED),
    ],
)
def test_cert_credential_grant_id_binding(
    use_mismatch_provider: bool,
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
    provider = _MismatchGrantProvider() if use_mismatch_provider else _grant_provider()
    preparer = ReferenceScopedAdaptedIntegrationEffectRequestPreparer()
    executor = ReferenceScopedAdaptedIntegrationEffectExecutor()
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_security_source=_sandbox_ok(),
        credential_broker=_broker(),
        credential_grant_provider=provider,
        effect_preparer=preparer,
        effect_executor=executor,
    )
    assert envelope.outcome is expected
    if expected is not ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED:
        assert executor.call_count == 0


@pytest.mark.parametrize(
    "sandbox_caps",
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
def test_cert_sandbox_attestation_matrix(sandbox_caps: SandboxSecurityCapabilities) -> None:
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
    preparer = ReferenceScopedAdaptedIntegrationEffectRequestPreparer()
    executor = ReferenceScopedAdaptedIntegrationEffectExecutor()
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_security_source=ReferenceScopedAdaptiveIntegrationSandboxSession(
            capabilities=sandbox_caps,
        ),
        credential_broker=_broker(),
        credential_grant_provider=_grant_provider(),
        effect_preparer=preparer,
        effect_executor=executor,
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.SANDBOX_SECURITY_UNSATISFIED
    assert executor.call_count == 0
