# © Artur Czarnecki. All rights reserved.

"""LLM-EXTERNAL-OPERATION-DIAGNOSTIC-INTEGRATION R2 qualification matrix."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmissionContext,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.attempt import ExternalOperationAttemptLifecycle
from intergrax.contracts.external_operations.execution_context import (
    ExternalOperationExecutionContext,
)
from intergrax.contracts.external_operations.failure import ExternalOperationFailureKind
from intergrax.contracts.external_operations.intent import (
    ExternalOperationIntent,
    ExternalOperationType,
    mint_external_operation_intent_id,
)
from intergrax.contracts.external_operations.evidence import ProviderExecutionOutcome
from intergrax.contracts.external_operations.provider import (
    ProviderPayloadBounds,
    ProviderRiskProfile,
)
from intergrax.contracts.external_operations.admission import OperationAdmissionDecision
from intergrax.contracts.external_operations.safety import (
    ExternalOperationAdmissionDeniedError,
    assert_no_secrets_in_audit_payload,
)
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessmentBuilder,
    DiagnosticFindingKind,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.external_operations.admission.execution_gate import (
    ExternalOperationExecutionGate,
)
from intergrax.runtime.external_operations.admission.local_admission import (
    PolicyExternalOperationAdmission,
)
from intergrax.runtime.external_operations.admission.provider_executor import (
    ContainedProviderExecutionError,
    execute_contained_provider_call,
    retry_contained_provider_attempt,
)
from intergrax.runtime.external_operations.diagnostic.failure_classification import (
    classify_provider_exception,
)
from intergrax.runtime.external_operations.diagnostic.runtime_event_recorder import (
    RuntimeEventExternalOperationFailureRecorder,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.prediction.external_operation_failure_history import (
    collect_external_operation_failure_history,
)
pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _AllowAllAdmission:
    def evaluate(self, intent: object, context: object) -> OperationAdmissionDecision:
        return OperationAdmissionDecision(verdict=OperationAdmissionVerdict.ALLOW, reason="test")

_SECRET = "super-secret-api-key-qualification"


def _intent(
    *,
    tenant_id: str = "tenant_a",
    task_id: str | None = None,
) -> ExternalOperationIntent:
    return ExternalOperationIntent(
        intent_id=mint_external_operation_intent_id(),
        tenant_id=tenant_id,
        task_id=task_id or mint_task_id(),
        operation_type=ExternalOperationType.CRM_UPDATE,
        target_resource="crm:connector",
        requested_by="operator",
        justification="update CRM record after diagnostic",
        created_at=datetime.now(timezone.utc),
    )


class _StubProvider:
    def __init__(self, *, fail: bool = False, exc: Exception | None = None) -> None:
        self._fail = fail
        self._exc = exc or PermissionError("forbidden")

    @property
    def provider_id(self) -> str:
        return "crm_provider"

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
            max_payload_bytes=4096,
            timeout_seconds=5.0,
            max_retries=0,
        )

    def execute_admitted(self, attempt: object) -> ProviderExecutionOutcome:
        if self._fail:
            raise self._exc
        return ProviderExecutionOutcome(status="success", safe_summary="ok")


def _execution_context(intent: ExternalOperationIntent) -> ExternalOperationExecutionContext:
    return ExternalOperationExecutionContext(
        execution_id=mint_execution_id(),
        attempt_id=mint_attempt_id(),
        tenant_id=intent.tenant_id,
        task_id=intent.task_id,
        operation_intent_id=intent.intent_id,
        provider_id="crm_provider",
        scope=intent.target_resource,
    )


def test_external_operation_uses_runtime_execution_id() -> None:
    intent = _intent()
    ctx = _execution_context(intent)
    gate = ExternalOperationExecutionGate(
        admission=_AllowAllAdmission(),
    )
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id=intent.tenant_id),
        provider_id="crm_provider",
    )
    admitted = admitted.bind_platform_execution(
        execution_id=ctx.execution_id,
        attempt_id=ctx.attempt_id,
    )
    assert admitted.execution_id == ctx.execution_id
    assert admitted.attempt_id == ctx.attempt_id


def test_operation_without_admission_never_executes() -> None:
    intent = _intent()
    gate = ExternalOperationExecutionGate(admission=PolicyExternalOperationAdmission())
    with pytest.raises(ExternalOperationAdmissionDeniedError):
        gate.admit_intent(
            intent,
            context=ExternalOperationAdmissionContext(tenant_id="tenant_b"),
        )


def test_provider_failure_creates_runtime_event() -> None:
    intent = _intent()
    ctx = _execution_context(intent)
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    recorder = RuntimeEventExternalOperationFailureRecorder(bus)
    gate = ExternalOperationExecutionGate(
        admission=_AllowAllAdmission(),
    )
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id=intent.tenant_id),
        provider_id="crm_provider",
    )
    provider = _StubProvider(fail=True)
    with pytest.raises(ContainedProviderExecutionError):
        execute_contained_provider_call(
            gate=gate,
            provider=provider,
            attempt=admitted,
            fn=lambda: provider.execute_admitted(admitted),  # noqa: ARG005
            execution_context=ctx,
            run_id=run_id,
            failure_recorder=recorder,
        )
    events = store.list_for_task(str(intent.task_id), tenant_id=intent.tenant_id)
    assert any(e.event_type is RuntimeEventType.EXTERNAL_OPERATION_FAILED for e in events)


def test_external_operation_failure_reaches_central_diagnostic_engine() -> None:
    intent = _intent()
    ctx = _execution_context(intent)
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    recorder = RuntimeEventExternalOperationFailureRecorder(bus)
    gate = ExternalOperationExecutionGate(
        admission=_AllowAllAdmission(),
    )
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id=intent.tenant_id),
        provider_id="crm_provider",
    )
    provider = _StubProvider(fail=True)
    with pytest.raises(ContainedProviderExecutionError):
        execute_contained_provider_call(
            gate=gate,
            provider=provider,
            attempt=admitted,
            fn=lambda: provider.execute_admitted(admitted),  # noqa: ARG005
            execution_context=ctx,
            run_id=run_id,
            failure_recorder=recorder,
        )
    reconstruction = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(intent.tenant_id, intent.task_id, run_id)
    assessment = DiagnosticAssessmentBuilder().assess(
        reconstruction,
        LifecycleAnomalyAnalyzer().analyze(reconstruction),
    )
    kinds = {f.kind for f in assessment.findings}
    assert DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED in kinds


def test_failure_boundary_points_to_external_operation_execution() -> None:
    intent = _intent()
    child_execution = mint_execution_id()
    ctx = ExternalOperationExecutionContext(
        execution_id=child_execution,
        attempt_id=mint_attempt_id(),
        tenant_id=intent.tenant_id,
        task_id=intent.task_id,
        operation_intent_id=intent.intent_id,
        provider_id="crm_provider",
        scope=intent.target_resource,
    )
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    recorder = RuntimeEventExternalOperationFailureRecorder(bus)
    gate = ExternalOperationExecutionGate(
        admission=_AllowAllAdmission(),
    )
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id=intent.tenant_id),
        provider_id="crm_provider",
    )
    provider = _StubProvider(fail=True)
    with pytest.raises(ContainedProviderExecutionError):
        execute_contained_provider_call(
            gate=gate,
            provider=provider,
            attempt=admitted,
            fn=lambda: provider.execute_admitted(admitted),  # noqa: ARG005
            execution_context=ctx,
            run_id=run_id,
            failure_recorder=recorder,
        )
    reconstruction = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(intent.tenant_id, intent.task_id, run_id)
    assessment = DiagnosticAssessmentBuilder().assess(
        reconstruction,
        LifecycleAnomalyAnalyzer().analyze(reconstruction),
    )
    finding = next(
        f for f in assessment.findings if f.kind is DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED
    )
    assert finding.execution_id == child_execution
    assert assessment.failure_boundary_analysis is not None
    failed = assessment.failure_boundary_analysis.topology.failed_boundaries
    assert any(b.execution_id == child_execution for b in failed)


def test_provider_failure_does_not_break_parent_execution() -> None:
    intent = _intent()
    ctx = _execution_context(intent)
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    recorder = RuntimeEventExternalOperationFailureRecorder(bus)
    gate = ExternalOperationExecutionGate(
        admission=_AllowAllAdmission(),
    )
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id=intent.tenant_id),
        provider_id="crm_provider",
    )
    parent_execution = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=ctx.attempt_id,
        execution_id=parent_execution,
    )
    try:
        provider = _StubProvider(fail=True)
        with pytest.raises(ContainedProviderExecutionError):
            execute_contained_provider_call(
                gate=gate,
                provider=provider,
                attempt=admitted,
                fn=lambda: provider.execute_admitted(admitted),
                execution_context=ctx,
                run_id=run_id,
                failure_recorder=recorder,
            )
    finally:
        reset_active_execution_identity(token)
    events = store.list_for_task(str(intent.task_id), tenant_id=intent.tenant_id)
    assert not any(e.event_type is RuntimeEventType.EXECUTION_FAILED for e in events)


def test_retry_creates_new_attempt_without_rewriting_history() -> None:
    intent = _intent()
    ctx = _execution_context(intent)
    gate = ExternalOperationExecutionGate(
        admission=_AllowAllAdmission(),
    )
    first = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id=intent.tenant_id),
        provider_id="crm_provider",
    )
    first = first.bind_platform_execution(
        execution_id=ctx.execution_id,
        attempt_id=ctx.attempt_id,
    )
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    recorder = RuntimeEventExternalOperationFailureRecorder(RuntimeEventBus(persistence=store))
    provider = _StubProvider(fail=True)
    with pytest.raises(ContainedProviderExecutionError):
        execute_contained_provider_call(
            gate=gate,
            provider=provider,
            attempt=first,
            fn=lambda: provider.execute_admitted(first),
            execution_context=ctx,
            run_id=run_id,
            failure_recorder=recorder,
        )
    executing = gate.begin_execution(
        first.bind_platform_execution(
            execution_id=ctx.execution_id,
            attempt_id=ctx.attempt_id,
        ),
    )
    failed = gate.complete_failure(executing)
    retry = retry_contained_provider_attempt(prior=failed, execution_context=ctx)
    assert retry.operation_attempt_id != first.operation_attempt_id
    assert retry.execution_id == ctx.execution_id
    assert retry.attempt_id == ctx.attempt_id


def test_external_operation_tenant_isolation() -> None:
    intent = _intent(tenant_id="tenant_a")
    gate = ExternalOperationExecutionGate(admission=PolicyExternalOperationAdmission())
    with pytest.raises(ExternalOperationAdmissionDeniedError):
        gate.admit_intent(
            intent,
            context=ExternalOperationAdmissionContext(tenant_id="tenant_b"),
        )


def test_external_operation_evidence_contains_no_credentials() -> None:
    intent = _intent()
    ctx = _execution_context(intent)
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    recorder = RuntimeEventExternalOperationFailureRecorder(RuntimeEventBus(persistence=store))

    class _LeakyProvider(_StubProvider):
        def execute_admitted(self, attempt: object) -> str:
            raise RuntimeError(f"boom {_SECRET}")

    gate = ExternalOperationExecutionGate(
        admission=_AllowAllAdmission(),
    )
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id=intent.tenant_id),
        provider_id="crm_provider",
    )
    leaky = _LeakyProvider(fail=True)
    with pytest.raises(ContainedProviderExecutionError) as exc_info:
        execute_contained_provider_call(
            gate=gate,
            provider=leaky,
            attempt=admitted,
            fn=lambda: leaky.execute_admitted(admitted),
            execution_context=ctx,
            run_id=run_id,
            failure_recorder=recorder,
        )
    blob = (
        exc_info.value.evidence.safe_summary,
        exc_info.value.failure.provider_failure.safe_message if exc_info.value.failure.provider_failure else "",
    )
    assert_no_secrets_in_audit_payload(blob)
    assert _SECRET not in str(blob)


def test_predictive_engine_consumes_external_operation_history() -> None:
    intent = _intent()
    ctx = _execution_context(intent)
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    recorder = RuntimeEventExternalOperationFailureRecorder(RuntimeEventBus(persistence=store))
    gate = ExternalOperationExecutionGate(
        admission=_AllowAllAdmission(),
    )
    admitted = gate.admit_intent(
        intent,
        context=ExternalOperationAdmissionContext(tenant_id=intent.tenant_id),
        provider_id="crm_provider",
    )
    provider = _StubProvider(fail=True, exc=TimeoutError("provider timeout"))
    with pytest.raises(ContainedProviderExecutionError):
        execute_contained_provider_call(
            gate=gate,
            provider=provider,
            attempt=admitted,
            fn=lambda: provider.execute_admitted(admitted),
            execution_context=ctx,
            run_id=run_id,
            failure_recorder=recorder,
        )
    history = collect_external_operation_failure_history(
        store,
        tenant_id=intent.tenant_id,
        task_id=intent.task_id,
        provider_id="crm_provider",
    )
    assert len(history) >= 1
    assert history[0].failure_kind is ExternalOperationFailureKind.TIMEOUT


def test_sap_exception_maps_to_remote_failure() -> None:
    class SAPException(Exception):
        pass

    assert (
        classify_provider_exception(SAPException("remote"))
        is ExternalOperationFailureKind.REMOTE_FAILURE
    )
