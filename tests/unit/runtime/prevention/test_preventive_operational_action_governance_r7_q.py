# © Artur Czarnecki. All rights reserved.

"""PREVENTIVE-OPERATIONAL-ACTION-GOVERNANCE R7 qualification matrix."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmissionContext,
    OperationAdmissionDecision,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.execution_context import (
    ExternalOperationExecutionContext,
)
from intergrax.contracts.external_operations.safety import (
    ExternalOperationAdmissionDeniedError,
    ExternalOperationApprovalRequiredError,
    assert_no_secrets_in_audit_payload,
)
from intergrax.contracts.preventive.actions.action_type import PreventiveActionType
from intergrax.contracts.preventive.actions.admission import PreventiveActionAdmissionContext
from intergrax.contracts.preventive.actions.outcome import (
    PreventiveActionObservedOutcome,
    PreventiveActionOutcomeEvaluation,
)
from intergrax.contracts.preventive.actions.safety import (
    assert_no_secrets_in_preventive_action_audit,
    assert_proposal_has_no_execution_surface,
)
from intergrax.contracts.preventive.actions.lifecycle import PreventiveActionLifecycleState
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
)
from intergrax.runtime.external_operations.diagnostic.runtime_event_recorder import (
    RuntimeEventExternalOperationFailureRecorder,
)
from intergrax.contracts.external_operations.evidence import ProviderExecutionOutcome
from intergrax.contracts.external_operations.provider import (
    ProviderPayloadBounds,
    ProviderRiskProfile,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.prediction.governance.analyzer_quality_store import (
    InMemoryPredictiveAnalyzerQualityStore,
)
from intergrax.runtime.prevention import PreventiveIntelligenceEngine, PreventiveAnalyzerRegistry
from intergrax.runtime.prevention.analyzers.crm_latency_preventive import CrmLatencyPreventiveAnalyzer
from intergrax.runtime.prevention.actions import (
    ConfigurationUpdatePreventiveActionProvider,
    GovernedPreventiveActionOrchestrator,
    PolicyPreventiveActionAdmissionGate,
    PreventiveActionOutcomeEngine,
    project_preventive_action_history,
)
from tests.unit.runtime.prevention.test_preventive_intelligence_governance_r6_q import (
    _analysis_input,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SECRET = "super-secret-api-key-qualification"
_ACTION_TYPE = f"{PreventiveActionType.CONFIGURATION_UPDATE.namespace}.{PreventiveActionType.CONFIGURATION_UPDATE.name}"


class _AllowAllExternalAdmission:
    def evaluate(self, intent: object, context: object) -> OperationAdmissionDecision:
        return OperationAdmissionDecision(verdict=OperationAdmissionVerdict.ALLOW, reason="test")


class _StubExternalProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    @property
    def provider_id(self) -> str:
        return "preventive.configuration_update"

    @property
    def version(self) -> str:
        return "1"

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"translate"})

    @property
    def tenant_scope(self) -> frozenset[str] | None:
        return None

    @property
    def risk_profile(self) -> ProviderRiskProfile:
        return ProviderRiskProfile.LOW

    @property
    def payload_bounds(self) -> ProviderPayloadBounds:
        return ProviderPayloadBounds(max_payload_bytes=2048, timeout_seconds=5.0, max_retries=0)

    def execute_admitted(self, attempt: object) -> ProviderExecutionOutcome:
        self.calls += 1
        if self._fail:
            raise PermissionError("forbidden")
        return ProviderExecutionOutcome(status="success", safe_summary="ok")


def _orchestrator(
    *,
    external_admission: object | None = None,
) -> GovernedPreventiveActionOrchestrator:
    provider = ConfigurationUpdatePreventiveActionProvider()
    ext_adm = external_admission or PolicyExternalOperationAdmission()
    return GovernedPreventiveActionOrchestrator(
        preventive_gate=PolicyPreventiveActionAdmissionGate(),
        external_gate=ExternalOperationExecutionGate(admission=ext_adm),
        external_admission=ext_adm,
        providers={provider.provider_id: provider},
    )


def _proposal(orchestrator: GovernedPreventiveActionOrchestrator, tenant_id: str = "tenant_a") -> object:
    return orchestrator.mint_proposal(
        tenant_id=tenant_id,
        risk_signal_refs=("prsig_r7",),
        recommendation_refs=("prrec_r7",),
        action_type=_ACTION_TYPE,
        target_resource="dev:crm:connector",
        justification="reduce synchronization batch after latency risk",
        confidence=0.92,
    )


def _approved_context(
    tenant_id: str = "tenant_a",
    *,
    approval_id: str = "apr_r7",
) -> PreventiveActionAdmissionContext:
    return PreventiveActionAdmissionContext(
        tenant_id=tenant_id,
        governance_approved=True,
        human_approval_granted=True,
        approval_id=approval_id,
        decision_id="dec_r7",
    )


def test_recommendation_does_not_execute_action() -> None:
    engine = PreventiveIntelligenceEngine(
        registry=PreventiveAnalyzerRegistry((CrmLatencyPreventiveAnalyzer(),)),
    )
    provider = _StubExternalProvider()
    rec = engine.recommend(_analysis_input()).recommendations[0]
    assert rec.safety.execution_allowed is False
    assert provider.calls == 0


def test_proposal_has_no_execute_surface() -> None:
    orch = _orchestrator()
    proposal = _proposal(orch)
    assert_proposal_has_no_execution_surface(proposal)
    assert not hasattr(proposal, "execute")


def test_denied_preventive_action_never_executes() -> None:
    orch = _orchestrator()
    proposal = _proposal(orch)
    provider = _StubExternalProvider()
    with pytest.raises(ExternalOperationAdmissionDeniedError):
        orch.attempt_execution(
            proposal,
            preventive_context=PreventiveActionAdmissionContext(
                tenant_id="tenant_a",
                governance_approved=False,
            ),
            task_id=mint_task_id(),
        )
    assert provider.calls == 0


def test_requires_approval_blocks_execution() -> None:
    orch = _orchestrator()
    proposal = _proposal(orch)
    with pytest.raises(ExternalOperationApprovalRequiredError):
        orch.attempt_execution(
            proposal,
            preventive_context=PreventiveActionAdmissionContext(
                tenant_id="tenant_a",
                governance_approved=True,
                human_approval_granted=False,
            ),
            task_id=mint_task_id(),
        )
    assert orch.lifecycle_by_proposal[proposal.proposal_id] is (
        PreventiveActionLifecycleState.WAITING_APPROVAL
    )


def test_approved_action_creates_external_operation_intent() -> None:
    orch = _orchestrator(external_admission=_AllowAllExternalAdmission())
    proposal = _proposal(orch)
    intent = orch.translate_proposal(proposal, task_id=mint_task_id())
    assert intent.intent_id.startswith("ext_op_intent_")
    assert intent.tenant_id == proposal.tenant_id


def test_preventive_action_uses_execution_runtime_identity() -> None:
    orch = _orchestrator(external_admission=_AllowAllExternalAdmission())
    proposal = _proposal(orch)
    task_id = mint_task_id()
    attempt = orch.attempt_execution(
        proposal,
        preventive_context=_approved_context(),
        task_id=task_id,
    )
    execution_id = mint_execution_id()
    attempt_id = mint_attempt_id()
    bound = attempt.bind_platform_execution(
        execution_id=execution_id,
        attempt_id=attempt_id,
    )
    assert bound.execution_id == execution_id
    assert bound.attempt_id == attempt_id


def test_failed_preventive_action_reaches_central_diagnostic_engine() -> None:
    intent_task = mint_task_id()
    orch = _orchestrator(external_admission=_AllowAllExternalAdmission())
    proposal = _proposal(orch)
    attempt = orch.attempt_execution(
        proposal,
        preventive_context=_approved_context(),
        task_id=intent_task,
    )
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    recorder = RuntimeEventExternalOperationFailureRecorder(bus)
    gate = ExternalOperationExecutionGate(admission=_AllowAllExternalAdmission())
    provider = _StubExternalProvider(fail=True)
    ctx = ExternalOperationExecutionContext(
        execution_id=mint_execution_id(),
        attempt_id=mint_attempt_id(),
        tenant_id=proposal.tenant_id,
        task_id=intent_task,
        operation_intent_id=attempt.intent.intent_id,
        provider_id=provider.provider_id,
        scope=proposal.target_resource,
    )
    with pytest.raises(ContainedProviderExecutionError):
        execute_contained_provider_call(
            gate=gate,
            provider=provider,
            attempt=attempt,
            fn=lambda: provider.execute_admitted(attempt),  # noqa: ARG005
            execution_context=ctx,
            run_id=run_id,
            failure_recorder=recorder,
        )
    events = store.list_for_task(str(intent_task), tenant_id=proposal.tenant_id)
    assert any(e.event_type is RuntimeEventType.EXTERNAL_OPERATION_FAILED for e in events)
    reconstruction = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(proposal.tenant_id, intent_task, run_id)
    assessment = DiagnosticAssessmentBuilder().assess(
        reconstruction,
        LifecycleAnomalyAnalyzer().analyze(reconstruction),
    )
    kinds = {f.kind for f in assessment.findings}
    assert DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED in kinds


def test_action_outcome_updates_prediction_quality() -> None:
    quality = InMemoryPredictiveAnalyzerQualityStore()
    engine = PreventiveActionOutcomeEngine(quality_store=quality)
    evaluation = PreventiveActionOutcomeEvaluation(
        proposal_id="pract_prop_deadbeef",
        tenant_id="tenant_a",
        analyzer_id="latency_trend",
        risk_signal_refs=("prsig_r7",),
        observed=PreventiveActionObservedOutcome.INCIDENT_AVOIDED,
        evidence_refs=("outcome:avoided",),
        evaluated_at=datetime.now(UTC),
    )
    engine.record(evaluation)
    profile = quality.get_profile(tenant_id="tenant_a", analyzer_id="latency_trend")
    assert profile.true_positive >= 1


def test_preventive_action_cannot_cross_tenant_boundary() -> None:
    orch = _orchestrator()
    proposal = _proposal(orch, tenant_id="tenant_a")
    with pytest.raises(ExternalOperationAdmissionDeniedError, match="tenant"):
        orch.attempt_execution(
            proposal,
            preventive_context=_approved_context(tenant_id="tenant_b"),
            task_id=mint_task_id(),
        )


def test_no_secret_leak_in_preventive_audit() -> None:
    orch = _orchestrator()
    proposal = _proposal(orch)
    with pytest.raises(ExternalOperationAdmissionDeniedError):
        orch.attempt_execution(
            proposal,
            preventive_context=PreventiveActionAdmissionContext(
                tenant_id="tenant_a",
                governance_approved=False,
            ),
            task_id=mint_task_id(),
        )
    record = orch.audit_records[-1]
    assert_no_secrets_in_preventive_action_audit(
        (record.admission_reason, record.proposal_id, record.action_type),
    )
    with pytest.raises(ValueError):
        assert_no_secrets_in_audit_payload(("sk-abcdefghijklmnopqrstuvwxyz123456",))


def test_read_model_preventive_action_history() -> None:
    orch = _orchestrator()
    proposal = _proposal(orch)
    with pytest.raises(ExternalOperationAdmissionDeniedError):
        orch.attempt_execution(
            proposal,
            preventive_context=PreventiveActionAdmissionContext(
                tenant_id="tenant_a",
                governance_approved=False,
            ),
            task_id=mint_task_id(),
        )
    views = project_preventive_action_history(tuple(orch.audit_records))
    assert len(views) == 1
    assert views[0].proposal_id == proposal.proposal_id
    assert views[0].outcome == "DENIED"
