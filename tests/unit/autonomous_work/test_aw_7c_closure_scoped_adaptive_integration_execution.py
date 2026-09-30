# © Artur Czarnecki. All rights reserved.

"""AW-7C-CLOSURE — canonical runtime-bound parent certification tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta

import pytest

from intergrax.autonomous_work.scoped_adaptive_integration_execution import (
    WorkerScopedAdaptiveIntegrationExecutionCoordinator,
)
from intergrax.capability_qualification.qualification_service import CapabilityQualificationService
from intergrax.contracts.autonomous_work.scoped_adaptive_integration_execution import (
    ScopedAdaptiveIntegrationExecutionOutcome,
    ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
    build_scoped_adaptive_integration_execution_handoff,
)
from intergrax.contracts.execution_identity import ExecutionId, mint_execution_id
from intergrax.contracts.execution_request import ExecutionRequest
from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
)
from intergrax.integrations.contracts.credential import CredentialRef, CredentialUseGrant
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationOperationEvidence,
    ScopedAdaptedIntegrationOperationPort,
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationOperationId,
)
from intergrax.integrations.qualification.reference_scoped_adaptive_integration_execution import (
    ReferenceExecutionBoundCredentialGrantProvider,
    ReferenceScopedAdaptedIntegrationOperation,
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
from intergrax.integrations.qualification.scoped_adaptive_integration_execution_runtime_delegate import (
    ScopedAdaptiveIntegrationExecutionRuntimeDelegate,
)
from intergrax.runtime.execution.canonical_intake_adapter import CanonicalExecutionRuntimeAdapter
from intergrax.runtime.sandbox.contracts import SandboxSecurityCapabilities
from tests.unit.autonomous_work.test_aw_7c_cert_scoped_adaptive_integration_execution import (
    _broker,
    _grant_provider,
    _sandbox_ok,
)
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


class _AltScopedAdaptedIntegrationOperation:
    """Replaceability proof — second operation implementation."""

    def execute(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        requested_operation: ScopedIntegrationAdaptationOperationId,
        execution_id: str,
        tenant_id: str,
    ) -> ScopedAdaptedIntegrationOperationEvidence:
        return ScopedAdaptedIntegrationOperationEvidence(
            evidence_ref=f"alt:{artifact.artifact_id}:{execution_id}",
            tenant_id=tenant_id,
            execution_id=execution_id,
            artifact_id=artifact.artifact_id,
            executed_operation=requested_operation,
        )


class _WriteGrantProvider:
    def resolve_grant(
        self,
        *,
        execution_id: ExecutionId,
        credential_grant_ref: str,
        tenant_id: str,
        provider_id: str,
        integration_id: str,
        requested_operation: ScopedIntegrationAdaptationOperationId,
    ) -> CredentialUseGrant:
        base = _grant_provider()
        return CredentialUseGrant(
            grant_id=credential_grant_ref,
            credential_ref=base.credential_ref,
            tenant_id=tenant_id,
            provider_id=provider_id,
            integration_id=integration_id,
            operation=reference_write_operation().value,
            execution_id=str(execution_id),
            target_scope=base.target_scope,
            expires_at=base.expires_at,
        )


class _WrongOpEvidenceOperation(ReferenceScopedAdaptedIntegrationOperation):
    def execute(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        requested_operation: ScopedIntegrationAdaptationOperationId,
        execution_id: str,
        tenant_id: str,
    ) -> ScopedAdaptedIntegrationOperationEvidence:
        wrong = reference_write_operation()
        return ScopedAdaptedIntegrationOperationEvidence(
            evidence_ref=f"bad:{artifact.artifact_id}",
            tenant_id=tenant_id,
            execution_id=execution_id,
            artifact_id=artifact.artifact_id,
            executed_operation=wrong,
        )


@dataclass
class _ExecutionOrderTrace:
    events: list[str]

    def record(self, name: str) -> None:
        self.events.append(name)


class _TracingGrantProvider:
    def __init__(
        self,
        inner: ReferenceExecutionBoundCredentialGrantProvider,
        trace: _ExecutionOrderTrace,
    ) -> None:
        self._inner = inner
        self._trace = trace

    def resolve_grant(
        self,
        *,
        execution_id: ExecutionId,
        credential_grant_ref: str,
        tenant_id: str,
        provider_id: str,
        integration_id: str,
        requested_operation: ScopedIntegrationAdaptationOperationId,
    ) -> CredentialUseGrant:
        self._trace.record("credential_grant_resolved")
        return self._inner.resolve_grant(
            execution_id=execution_id,
            credential_grant_ref=credential_grant_ref,
            tenant_id=tenant_id,
            provider_id=provider_id,
            integration_id=integration_id,
            requested_operation=requested_operation,
        )


class _TracingSandboxSession:
    def __init__(
        self,
        inner: ReferenceScopedAdaptiveIntegrationSandboxSession,
        trace: _ExecutionOrderTrace,
    ) -> None:
        self._inner = inner
        self._trace = trace

    def security_capabilities(self) -> SandboxSecurityCapabilities:
        self._trace.record("sandbox_capabilities_obtained")
        return self._inner.security_capabilities()


class _TracingOperation(ReferenceScopedAdaptedIntegrationOperation):
    trace: _ExecutionOrderTrace

    def execute(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        requested_operation: ScopedIntegrationAdaptationOperationId,
        execution_id: str,
        tenant_id: str,
    ) -> ScopedAdaptedIntegrationOperationEvidence:
        self.trace.record("operation_invoked")
        return super().execute(
            artifact=artifact,
            requested_operation=requested_operation,
            execution_id=execution_id,
            tenant_id=tenant_id,
        )


def _canonical_stack(
    *,
    operation_port: ScopedAdaptedIntegrationOperationPort | None = None,
    grant_provider: object | None = None,
    trace: _ExecutionOrderTrace | None = None,
) -> tuple[object, ScopedAdaptiveIntegrationExecutionRuntimeDelegate, ScopedAdaptedIntegrationOperationPort]:
    broker = _broker()
    sandbox = _sandbox_ok()
    provider = grant_provider or _grant_provider()
    op = operation_port or ReferenceScopedAdaptedIntegrationOperation()
    if trace is not None:
        provider = _TracingGrantProvider(_grant_provider(), trace)
        sandbox = _TracingSandboxSession(sandbox, trace)
        if type(op) is ReferenceScopedAdaptedIntegrationOperation:
            traced = _TracingOperation()
            traced.trace = trace
            op = traced
    intake, delegate = build_scoped_adaptive_integration_canonical_execution_intake(
        credential_broker=broker,
        credential_grant_provider=provider,
        sandbox_security_source=sandbox,
        operation_port=op,
    )
    assert type(intake) is CanonicalExecutionRuntimeAdapter
    return intake, delegate, op


@pytest.mark.asyncio
async def test_closure_canonical_e2e_execution_id_continuity() -> None:
    preparation = _prepare()
    intake, delegate, _op = _canonical_stack()
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert result.run_id is not None
    assert delegate.execute_calls == 1


@pytest.mark.asyncio
async def test_closure_delegate_without_runtime_active_id_fails_closed() -> None:
    delegate = ScopedAdaptiveIntegrationExecutionRuntimeDelegate(
        credential_broker=_broker(),
        credential_grant_provider=_grant_provider(),
        sandbox_security_source=_sandbox_ok(),
        operation_port=ReferenceScopedAdaptedIntegrationOperation(),
    )
    preparation = _prepare()
    qual_request = preparation.qualification_request
    assert qual_request is not None
    decision = CapabilityQualificationService(
        (ReferenceScopedIntegrationQualificationProvider(),),
    ).qualify(qual_request)
    handoff = build_scoped_adaptive_integration_execution_handoff(
        preparation=preparation,
        qualification_request=qual_request,
        accepted_qualification=decision,
        requested_operation=reference_read_operation(),
        execution_idempotency_key="no-runtime",
    )
    envelope = await delegate.execute(
        ExecutionRequest(
            input=handoff,
            output_type=ScopedAdaptiveIntegrationExecutionRuntimeEnvelope,
        ),
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED


@pytest.mark.asyncio
async def test_closure_operation_binding_read_pass() -> None:
    preparation = _prepare()
    intake, _delegate, op = _canonical_stack()
    assert type(op) is ReferenceScopedAdaptedIntegrationOperation
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert op.last_operation == reference_read_operation()


def test_closure_operation_binding_write_pass() -> None:
    preparation = _prepare()
    qual_request = preparation.qualification_request
    assert qual_request is not None
    decision = CapabilityQualificationService(
        (ReferenceScopedIntegrationQualificationProvider(),),
    ).qualify(qual_request)
    handoff = build_scoped_adaptive_integration_execution_handoff(
        preparation=preparation,
        qualification_request=qual_request,
        accepted_qualification=decision,
        requested_operation=reference_read_operation(),
        execution_idempotency_key="write-op",
    )
    handoff = replace(
        handoff,
        requested_operation=reference_write_operation(),
        permitted_operations=(reference_read_operation(), reference_write_operation()),
    )
    execution_id = mint_execution_id()
    operation = ReferenceScopedAdaptedIntegrationOperation()
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_security_source=_sandbox_ok(),
        credential_broker=_broker(),
        credential_grant_provider=_grant_provider(),
        operation_port=operation,
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert operation.last_operation == reference_write_operation()


def test_closure_handoff_read_grant_write_rejected() -> None:
    preparation = _prepare()
    qual_request = preparation.qualification_request
    assert qual_request is not None
    decision = CapabilityQualificationService(
        (ReferenceScopedIntegrationQualificationProvider(),),
    ).qualify(qual_request)
    handoff = build_scoped_adaptive_integration_execution_handoff(
        preparation=preparation,
        qualification_request=qual_request,
        accepted_qualification=decision,
        requested_operation=reference_read_operation(),
        execution_idempotency_key="op-mismatch",
    )
    execution_id = mint_execution_id()
    operation = ReferenceScopedAdaptedIntegrationOperation()
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_security_source=_sandbox_ok(),
        credential_broker=_broker(),
        credential_grant_provider=_WriteGrantProvider(),
        operation_port=operation,
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED
    assert operation.last_operation is None


def test_closure_operation_evidence_mismatch_rejected() -> None:
    preparation = _prepare()
    qual_request = preparation.qualification_request
    assert qual_request is not None
    decision = CapabilityQualificationService(
        (ReferenceScopedIntegrationQualificationProvider(),),
    ).qualify(qual_request)
    handoff = build_scoped_adaptive_integration_execution_handoff(
        preparation=preparation,
        qualification_request=qual_request,
        accepted_qualification=decision,
        requested_operation=reference_read_operation(),
        execution_idempotency_key="evidence-mismatch",
    )
    execution_id = mint_execution_id()
    operation = _WrongOpEvidenceOperation()
    envelope = execute_reference_scoped_adaptive_integration(
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_security_source=_sandbox_ok(),
        credential_broker=_broker(),
        credential_grant_provider=_grant_provider(),
        operation_port=operation,
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED


@pytest.mark.asyncio
async def test_closure_replaceable_operation_port() -> None:
    preparation = _prepare()
    intake, _delegate, _op = _canonical_stack(operation_port=_AltScopedAdaptedIntegrationOperation())
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert result.operation_output is not None
    assert result.operation_output.evidence_ref.startswith("alt:")


@pytest.mark.asyncio
async def test_closure_execution_order_observable() -> None:
    trace = _ExecutionOrderTrace(events=[])
    preparation = _prepare()
    intake, _delegate, _op = _canonical_stack(trace=trace)
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert trace.events.index("sandbox_capabilities_obtained") < trace.events.index(
        "credential_grant_resolved",
    )
    assert trace.events.index("credential_grant_resolved") < trace.events.index(
        "operation_invoked",
    )
