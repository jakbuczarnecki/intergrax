# © Artur Czarnecki. All rights reserved.

"""AW-7C-CLOSURE-R1 — typed effect preparation, late credential resolution, canonical executor."""

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
from intergrax.integrations.contracts.credential import (
    CredentialScopeMismatchError,
    CredentialUseGrant,
)
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedAdaptedIntegrationEffectRequest,
    ScopedAdaptedIntegrationEffectRequestPort,
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationOperationId,
)
from intergrax.integrations.credentials.broker import ScopedCredentialBroker
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
_HOST_B = NetworkEgressHost(scheme="https", hostname="b.example.com", port=443)
_ALLOWLIST = NetworkEgressAllowlist(hosts=(_HOST_A,))
_WIDENED_ALLOWLIST = NetworkEgressAllowlist(hosts=(_HOST_A, _HOST_B))


class _CustomPreparerSameContract(ReferenceScopedAdaptedIntegrationEffectRequestPreparer):
    """Replaceable preparer emitting valid requests (pluginability)."""

    marker = "custom-preparer"


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
            grant_id=base.grant_id,
            credential_ref=base.credential_ref,
            tenant_id=base.tenant_id,
            provider_id=base.provider_id,
            integration_id=base.integration_id,
            operation=reference_write_operation().value,
            execution_id=str(execution_id),
            target_scope=base.target_scope,
            expires_at=base.expires_at,
        )


class _BrokenEffectExecutor(ReferenceScopedAdaptedIntegrationEffectExecutor):
    def execute(self, ingress: object) -> object:
        from intergrax.integrations.contracts.scoped_integration_adaptation import (
            ScopedAdaptedIntegrationOperationEvidence,
        )

        effect_request = ingress.effect_request  # type: ignore[attr-defined]
        wrong = reference_write_operation()
        return ScopedAdaptedIntegrationOperationEvidence(
            evidence_ref=f"bad:{effect_request.artifact_id}",
            tenant_id=effect_request.tenant_id,
            execution_id=effect_request.execution_id,
            artifact_id=effect_request.artifact_id,
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


class _TracingPreparer(ReferenceScopedAdaptedIntegrationEffectRequestPreparer):
    def __init__(self, trace: _ExecutionOrderTrace) -> None:
        super().__init__()
        self._trace = trace

    def prepare(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        requested_operation: ScopedIntegrationAdaptationOperationId,
        execution_id: str,
        tenant_id: str,
        admitted_network_allowlist: NetworkEgressAllowlist,
    ) -> ScopedAdaptedIntegrationEffectRequest:
        self._trace.record("effect_preparer_called")
        return super().prepare(
            artifact=artifact,
            requested_operation=requested_operation,
            execution_id=execution_id,
            tenant_id=tenant_id,
            admitted_network_allowlist=admitted_network_allowlist,
        )


class _TracingBroker:
    def __init__(self, inner: ScopedCredentialBroker, trace: _ExecutionOrderTrace) -> None:
        self._inner = inner
        self._trace = trace
        self.resolve_scoped_calls = 0

    def resolve_scoped(self, grant: CredentialUseGrant, scope: object) -> object:
        self.resolve_scoped_calls += 1
        self._trace.record("credential_broker_resolved")
        return self._inner.resolve_scoped(grant, scope)


class _TracingExecutor(ReferenceScopedAdaptedIntegrationEffectExecutor):
    def __init__(self, trace: _ExecutionOrderTrace) -> None:
        super().__init__()
        self._trace = trace

    def execute(self, ingress: object) -> object:
        self._trace.record("canonical_effect_executor_called")
        return super().execute(ingress)  # type: ignore[arg-type]


class _CountingBroker:
    def __init__(self, inner: ScopedCredentialBroker) -> None:
        self._inner = inner
        self.resolve_scoped_calls = 0

    def resolve_scoped(self, grant: CredentialUseGrant, scope: object) -> object:
        self.resolve_scoped_calls += 1
        return self._inner.resolve_scoped(grant, scope)


class _CountingExecutor(ReferenceScopedAdaptedIntegrationEffectExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.execute_calls = 0

    def execute(self, ingress: object) -> object:
        self.execute_calls += 1
        return super().execute(ingress)  # type: ignore[arg-type]


class _FailingPreparer:
    def prepare(self, **kwargs: object) -> ScopedAdaptedIntegrationEffectRequest:
        raise RuntimeError("preparer unavailable")


class _InvalidEffectPreparer(ReferenceScopedAdaptedIntegrationEffectRequestPreparer):
    def __init__(self, mutator) -> None:
        super().__init__()
        self._mutator = mutator

    def prepare(
        self,
        *,
        artifact: ScopedIntegrationAdaptationArtifact,
        requested_operation: ScopedIntegrationAdaptationOperationId,
        execution_id: str,
        tenant_id: str,
        admitted_network_allowlist: NetworkEgressAllowlist,
    ) -> ScopedAdaptedIntegrationEffectRequest:
        request = super().prepare(
            artifact=artifact,
            requested_operation=requested_operation,
            execution_id=execution_id,
            tenant_id=tenant_id,
            admitted_network_allowlist=admitted_network_allowlist,
        )
        return self._mutator(request)


def _canonical_stack(
    *,
    effect_preparer: ScopedAdaptedIntegrationEffectRequestPort | None = None,
    grant_provider: object | None = None,
    trace: _ExecutionOrderTrace | None = None,
    broker: ScopedCredentialBroker | _TracingBroker | _CountingBroker | None = None,
    effect_executor: ReferenceScopedAdaptedIntegrationEffectExecutor | None = None,
) -> tuple[object, ScopedAdaptiveIntegrationExecutionRuntimeDelegate]:
    from intergrax.runtime.execution.runtime import ExecutionRuntime

    base_broker = broker or _broker()
    sandbox = _sandbox_ok()
    provider = grant_provider or _grant_provider()
    preparer = effect_preparer or ReferenceScopedAdaptedIntegrationEffectRequestPreparer()
    executor = effect_executor or ReferenceScopedAdaptedIntegrationEffectExecutor()
    if trace is not None:
        provider = _TracingGrantProvider(_grant_provider(), trace)
        sandbox = _TracingSandboxSession(sandbox, trace)
        preparer = _TracingPreparer(trace)
        if broker is None:
            base_broker = _TracingBroker(_broker(), trace)
        executor = _TracingExecutor(trace)
    delegate = ScopedAdaptiveIntegrationExecutionRuntimeDelegate(
        credential_broker=base_broker,
        credential_grant_provider=provider,
        sandbox_security_source=sandbox,
        effect_preparer=preparer,
        effect_executor=executor,
    )
    intake = CanonicalExecutionRuntimeAdapter(ExecutionRuntime(delegate))
    assert type(intake) is CanonicalExecutionRuntimeAdapter
    return intake, delegate


def _direct_execute(
    handoff: object,
    *,
    preparer: ScopedAdaptedIntegrationEffectRequestPort | None = None,
    grant_provider: object | None = None,
    broker: ScopedCredentialBroker | _CountingBroker | None = None,
    executor: ReferenceScopedAdaptedIntegrationEffectExecutor | None = None,
) -> ScopedAdaptiveIntegrationExecutionRuntimeEnvelope:
    execution_id = mint_execution_id()
    return execute_reference_scoped_adaptive_integration(
        handoff=handoff,
        execution_id=execution_id,
        tenant_id=_TENANT,
        sandbox_security_source=_sandbox_ok(),
        credential_broker=broker or _broker(),
        credential_grant_provider=grant_provider or _grant_provider(),
        effect_preparer=preparer or ReferenceScopedAdaptedIntegrationEffectRequestPreparer(),
        effect_executor=executor or ReferenceScopedAdaptedIntegrationEffectExecutor(),
    )


@pytest.mark.asyncio
async def test_closure_canonical_e2e_execution_id_continuity() -> None:
    preparation = _prepare()
    intake, delegate = _canonical_stack()
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
        effect_preparer=ReferenceScopedAdaptedIntegrationEffectRequestPreparer(),
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
    intake, delegate = _canonical_stack()
    preparer = delegate.effect_preparer
    assert type(preparer) is ReferenceScopedAdaptedIntegrationEffectRequestPreparer
    dispatch = _dispatch_stack(intake)
    coordinator = WorkerScopedAdaptiveIntegrationExecutionCoordinator(
        qualification_service=CapabilityQualificationService(
            (ReferenceScopedIntegrationQualificationProvider(),),
        ),
        dispatch_service=dispatch,
    )
    result = await coordinator.execute(_p4_request(preparation))
    assert result.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert preparer.last_requested_operation == reference_read_operation()


def test_closure_write_operation_outside_artifact_scope_rejected() -> None:
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
    broker = _CountingBroker(_broker())
    executor = _CountingExecutor()
    envelope = _direct_execute(handoff, broker=broker, executor=executor)
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED
    assert envelope.error_detail == "effect request operation outside artifact scope"
    assert broker.resolve_scoped_calls == 0
    assert executor.execute_calls == 0


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
    broker = _CountingBroker(_broker())
    executor = _CountingExecutor()
    envelope = _direct_execute(
        handoff,
        grant_provider=_WriteGrantProvider(),
        broker=broker,
        executor=executor,
    )
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.CREDENTIAL_DENIED
    assert broker.resolve_scoped_calls == 0
    assert executor.execute_calls == 0


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
    envelope = _direct_execute(handoff, executor=_BrokenEffectExecutor())
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED


@pytest.mark.asyncio
async def test_closure_replaceable_effect_preparer() -> None:
    preparation = _prepare()
    intake, _delegate = _canonical_stack(effect_preparer=_CustomPreparerSameContract())
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
    assert result.operation_output.evidence_ref.startswith("ref-effect:")


@pytest.mark.asyncio
async def test_closure_execution_order_observable() -> None:
    trace = _ExecutionOrderTrace(events=[])
    preparation = _prepare()
    intake, _delegate = _canonical_stack(trace=trace)
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
        "effect_preparer_called",
    )
    assert trace.events.index("effect_preparer_called") < trace.events.index(
        "credential_grant_resolved",
    )
    assert trace.events.index("credential_grant_resolved") < trace.events.index(
        "credential_broker_resolved",
    )
    assert trace.events.index("credential_broker_resolved") < trace.events.index(
        "canonical_effect_executor_called",
    )


@pytest.mark.parametrize(
    ("label", "mutator"),
    [
        ("tenant", lambda r: replace(r, tenant_id="tenant-other")),
        ("artifact_id", lambda r: replace(r, artifact_id="other-artifact")),
        ("artifact_fp", lambda r: replace(r, artifact_fingerprint="sha256:dead")),
        ("provider", lambda r: replace(r, provider_id="other-provider")),
        ("resource", lambda r: replace(r, resource_scope="other-resource")),
        ("operation", lambda r: replace(r, requested_operation=reference_write_operation())),
        ("network", lambda r: replace(r, network_allowlist=_WIDENED_ALLOWLIST)),
        ("target", lambda r: replace(r, target_scope=_WIDENED_ALLOWLIST)),
    ],
)
def test_closure_invalid_effect_request_zero_resolution(
    label: str,
    mutator: object,
) -> None:
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
        execution_idempotency_key=f"bad-effect-{label}",
    )
    broker = _CountingBroker(_broker())
    executor = _CountingExecutor()
    envelope = _direct_execute(
        handoff,
        preparer=_InvalidEffectPreparer(mutator),
        broker=broker,
        executor=executor,
    )
    assert envelope.outcome is not ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED
    assert broker.resolve_scoped_calls == 0
    assert executor.execute_calls == 0


def test_closure_preparer_failure_zero_resolution() -> None:
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
        execution_idempotency_key="preparer-fail",
    )
    broker = _CountingBroker(_broker())
    executor = _CountingExecutor()
    envelope = _direct_execute(handoff, preparer=_FailingPreparer(), broker=broker, executor=executor)
    assert envelope.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTION_FAILED
    assert broker.resolve_scoped_calls == 0
    assert executor.execute_calls == 0


@pytest.mark.parametrize(
    ("kwargs_patch"),
    [
        ({"credential_grant_ref": "wrong-grant"}),
        ({"tenant_id": "tenant-b"}),
        ({"provider_id": "provider-b"}),
        ({"integration_id": "integration-b"}),
    ],
)
def test_closure_grant_provider_authoritative_rejection(kwargs_patch: dict[str, str]) -> None:
    provider = _grant_provider()
    execution_id = mint_execution_id()
    base_kwargs = {
        "execution_id": execution_id,
        "credential_grant_ref": provider.grant_id,
        "tenant_id": provider.tenant_id,
        "provider_id": provider.provider_id,
        "integration_id": provider.integration_id,
        "requested_operation": reference_read_operation(),
    }
    base_kwargs.update(kwargs_patch)
    with pytest.raises(CredentialScopeMismatchError):
        provider.resolve_grant(**base_kwargs)


def test_closure_grant_provider_returns_authoritative_values() -> None:
    provider = _grant_provider()
    execution_id = mint_execution_id()
    grant = provider.resolve_grant(
        execution_id=execution_id,
        credential_grant_ref=provider.grant_id,
        tenant_id=provider.tenant_id,
        provider_id=provider.provider_id,
        integration_id=provider.integration_id,
        requested_operation=reference_read_operation(),
    )
    assert grant.grant_id == provider.grant_id
    assert grant.tenant_id == provider.tenant_id
    assert grant.provider_id == provider.provider_id
    assert grant.integration_id == provider.integration_id
    assert grant.execution_id == str(execution_id)
    assert grant.operation == reference_read_operation().value
