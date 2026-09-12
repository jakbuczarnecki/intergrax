# © Artur Czarnecki. All rights reserved.

"""AUTONOMOUS-ENTERPRISE-SELF-HEALING-STRATEGY R1 qualification matrix."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.external_operations.admission import (
    OperationAdmissionDecision,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.evidence import ProviderExecutionOutcome
from intergrax.contracts.external_operations.execution_context import (
    ExternalOperationExecutionContext,
)
from intergrax.contracts.external_operations.provider import (
    ProviderPayloadBounds,
    ProviderRiskProfile,
)
from intergrax.contracts.external_operations.safety import (
    ExternalOperationAdmissionDeniedError,
    ExternalOperationApprovalRequiredError,
)
from intergrax.contracts.self_healing import (
    SelfHealingContext,
    SelfHealingDecision,
    SelfHealingDiagnosticInvestigation,
    SelfHealingHistoricalOutcome,
    SelfHealingOperationDescriptor,
    SelfHealingPolicyConstraints,
    SelfHealingPredictiveSignal,
    SelfHealingProposedAction,
    SelfHealingStrategy,
    SelfHealingStrategyDescriptor,
    SelfHealingStrategyEvaluationStatus,
    assert_decision_has_no_execution_surface,
    assert_strategy_has_no_execution_surface,
    mint_self_healing_decision_id,
)
from intergrax.contracts.self_healing.governance import SelfHealingAdmissionContext
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessmentBuilder,
    DiagnosticFindingKind,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.event_bus import RuntimeEventBus
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
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.self_healing import (
    GovernedSelfHealingOrchestrator,
    InMemorySelfHealingStrategyRegistry,
    PlatformSelfHealingActionProvider,
    PolicySelfHealingAdmissionGate,
    SelfHealingDecisionEngine,
    platform_default_strategies,
    project_self_healing_history,
    resolve_strategies_for_context,
)
from intergrax.runtime.self_healing.defaults.generic_retry import GenericRetryAdjustmentStrategy

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RETRY_CAPABILITY = "execution.retry_pressure"
_ACTION = "self_healing.retry.adjustment"


class _AllowAllExternalAdmission:
    def evaluate(self, intent: object, context: object) -> OperationAdmissionDecision:
        return OperationAdmissionDecision(verdict=OperationAdmissionVerdict.ALLOW, reason="test")


class _StubExternalProvider:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    @property
    def provider_id(self) -> str:
        return "self_healing.platform.generic"

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


class _EnterpriseCrmRetryStrategy:
    strategy_id = "crm.batch.optimization"
    version = "1"

    @property
    def descriptor(self) -> SelfHealingStrategyDescriptor:
        return SelfHealingStrategyDescriptor(
            strategy_id=self.strategy_id,
            version=self.version,
            owner="application",
            capabilities=(_RETRY_CAPABILITY,),
            tenant_scope=frozenset({"tenant_a"}),
            priority=50,
            specificity=10,
            timeout_seconds=1.0,
            resource_budget_tokens=128,
        )

    def evaluate(self, context: SelfHealingContext) -> SelfHealingDecision | None:
        op = context.available_operations[0]
        return SelfHealingDecision(
            decision_id=mint_self_healing_decision_id(),
            strategy_id=self.strategy_id,
            confidence=0.88,
            proposed_actions=(
                SelfHealingProposedAction(
                    action_type=_ACTION,
                    target_resource=op.target_resource,
                    operation_kind=op.operation_kind,
                    provider_id=op.provider_id,
                    rationale="crm-specific batch optimization",
                ),
            ),
            evidence_refs=context.diagnostic_investigation.evidence_refs,
            justification="application plugin override",
            required_approval=False,
        )


class _ExplodingStrategy:
    strategy_id = "vendor.exploding"
    version = "1"

    @property
    def descriptor(self) -> SelfHealingStrategyDescriptor:
        return SelfHealingStrategyDescriptor(
            strategy_id=self.strategy_id,
            version=self.version,
            owner="vendor",
            capabilities=(_RETRY_CAPABILITY,),
            tenant_scope=None,
            priority=100,
            specificity=99,
            timeout_seconds=1.0,
            resource_budget_tokens=64,
        )

    def evaluate(self, context: SelfHealingContext) -> SelfHealingDecision | None:
        raise RuntimeError("plugin exploded")


def _sample_context(*, tenant_id: str = "tenant_a") -> SelfHealingContext:
    return SelfHealingContext(
        tenant_id=tenant_id,
        diagnostic_investigation=SelfHealingDiagnosticInvestigation(
            investigation_id="inv_sh_r1",
            problem_id="prob_sh_r1",
            tenant_id=tenant_id,
            evidence_refs=("ev_sh_1",),
            capability_tags=(_RETRY_CAPABILITY,),
        ),
        predictive_signals=(
            SelfHealingPredictiveSignal(
                tenant_id=tenant_id,
                signal_id="sig_1",
                signal_type=_RETRY_CAPABILITY,
            ),
        ),
        historical_outcomes=(
            SelfHealingHistoricalOutcome(tenant_id=tenant_id, strategy_id="platform.default.generic_retry_adjustment"),
        ),
        available_operations=(
            SelfHealingOperationDescriptor(
                operation_kind="retry.backoff.adjust",
                target_resource="dev:connector:sync",
                provider_id="self_healing.platform.generic",
            ),
        ),
        constraints=SelfHealingPolicyConstraints(production_target=False),
    )


def _registry_with_defaults() -> InMemorySelfHealingStrategyRegistry:
    registry = InMemorySelfHealingStrategyRegistry()
    for strategy in platform_default_strategies():
        registry.register(strategy)
    return registry


def _orchestrator(*, external_admission: object | None = None) -> GovernedSelfHealingOrchestrator:
    provider = PlatformSelfHealingActionProvider()
    ext_adm = external_admission or PolicyExternalOperationAdmission()
    return GovernedSelfHealingOrchestrator(
        admission_gate=PolicySelfHealingAdmissionGate(),
        external_gate=ExternalOperationExecutionGate(admission=ext_adm),
        external_admission=ext_adm,
        providers={provider.provider_id: provider},
    )


def _approved_context(tenant_id: str = "tenant_a") -> SelfHealingAdmissionContext:
    return SelfHealingAdmissionContext(
        tenant_id=tenant_id,
        governance_approved=True,
        human_approval_granted=True,
        approval_id="apr_sh_r1",
        decision_id="gov_dec_sh_r1",
    )


def test_custom_strategy_implements_platform_contract() -> None:
    plugin = _EnterpriseCrmRetryStrategy()
    assert isinstance(plugin, SelfHealingStrategy)
    assert_strategy_has_no_execution_surface(plugin)
    decision = plugin.evaluate(_sample_context())
    assert decision is not None
    assert_decision_has_no_execution_surface(decision)


def test_platform_default_strategy_available() -> None:
    registry = _registry_with_defaults()
    strategy = registry.resolve("platform.default.generic_retry_adjustment")
    assert strategy is not None
    assert isinstance(strategy, GenericRetryAdjustmentStrategy)


def test_application_strategy_overrides_default() -> None:
    registry = _registry_with_defaults()
    registry.register(_EnterpriseCrmRetryStrategy())
    ordered = resolve_strategies_for_context(registry.list_available(tenant_id="tenant_a"), _sample_context())
    assert ordered[0].strategy_id == "crm.batch.optimization"


def test_failed_strategy_does_not_break_engine() -> None:
    registry = _registry_with_defaults()
    registry.register(_ExplodingStrategy())
    engine = SelfHealingDecisionEngine(registry)
    results = engine.select_and_evaluate(_sample_context())
    assert results[0].status is SelfHealingStrategyEvaluationStatus.STRATEGY_FAILED
    assert any(r.status is SelfHealingStrategyEvaluationStatus.DECISION for r in results)


def test_strategy_cannot_bypass_governance() -> None:
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    results = engine.select_and_evaluate(_sample_context())
    decision = next(r.decision for r in results if r.decision is not None)
    orch = _orchestrator()
    with pytest.raises(ExternalOperationAdmissionDeniedError):
        orch.attempt_execution(
            decision,
            tenant_id="tenant_a",
            context=_sample_context(),
            admission_context=SelfHealingAdmissionContext(
                tenant_id="tenant_a",
                governance_approved=False,
            ),
            task_id=mint_task_id(),
        )


def test_strategy_uses_external_operation_spine() -> None:
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    decision = engine.select_and_evaluate(_sample_context())[-1].decision
    assert decision is not None
    orch = _orchestrator(external_admission=_AllowAllExternalAdmission())
    attempt = orch.attempt_execution(
        decision,
        tenant_id="tenant_a",
        context=_sample_context(),
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert attempt.intent.intent_id.startswith("ext_op_intent_")


def test_failed_self_healing_creates_central_evidence() -> None:
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    decision = engine.select_and_evaluate(_sample_context())[-1].decision
    assert decision is not None
    intent_task = mint_task_id()
    orch = _orchestrator(external_admission=_AllowAllExternalAdmission())
    attempt = orch.attempt_execution(
        decision,
        tenant_id="tenant_a",
        context=_sample_context(),
        admission_context=_approved_context(),
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
        tenant_id="tenant_a",
        task_id=intent_task,
        operation_intent_id=attempt.intent.intent_id,
        provider_id=provider.provider_id,
        scope=decision.proposed_actions[0].target_resource,
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
    events = store.list_for_task(str(intent_task), tenant_id="tenant_a")
    assert any(e.event_type is RuntimeEventType.EXTERNAL_OPERATION_FAILED for e in events)
    reconstruction = ExecutionReconstructor(
        runtime_events=store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution("tenant_a", intent_task, run_id)
    assessment = DiagnosticAssessmentBuilder().assess(
        reconstruction,
        LifecycleAnomalyAnalyzer().analyze(reconstruction),
    )
    kinds = {f.kind for f in assessment.findings}
    assert DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED in kinds


def test_strategy_is_tenant_scoped() -> None:
    registry = InMemorySelfHealingStrategyRegistry()
    scoped = _EnterpriseCrmRetryStrategy()
    registry.register(scoped)
    assert registry.list_available(tenant_id="tenant_a") == (scoped,)
    assert registry.list_available(tenant_id="tenant_b") == ()


def test_strategy_cannot_emit_execution_directly() -> None:
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    decision = engine.select_and_evaluate(_sample_context())[-1].decision
    assert decision is not None
    assert_decision_has_no_execution_surface(decision)
    provider = _StubExternalProvider()
    assert provider.calls == 0


def test_self_healing_history_projects_to_investigation_read_model() -> None:
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    decision = engine.select_and_evaluate(_sample_context())[-1].decision
    assert decision is not None
    orch = _orchestrator(external_admission=_AllowAllExternalAdmission())
    orch.attempt_execution(
        decision,
        tenant_id="tenant_a",
        context=_sample_context(),
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    views = project_self_healing_history(
        tuple(orch.audit_records),
        justification_by_decision={decision.decision_id: decision.justification},
    )
    assert views[0].strategy_id == decision.strategy_id
    assert views[0].outcome == "SUCCEEDED"


def test_high_confidence_still_requires_governance_when_marked() -> None:
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    decision = engine.select_and_evaluate(_sample_context())[-1].decision
    assert decision is not None
    orch = _orchestrator()
    with pytest.raises(ExternalOperationApprovalRequiredError):
        orch.attempt_execution(
            decision,
            tenant_id="tenant_a",
            context=SelfHealingContext(
                tenant_id="tenant_a",
                diagnostic_investigation=_sample_context().diagnostic_investigation,
                predictive_signals=_sample_context().predictive_signals,
                historical_outcomes=_sample_context().historical_outcomes,
                available_operations=_sample_context().available_operations,
                constraints=SelfHealingPolicyConstraints(production_target=True),
            ),
            admission_context=SelfHealingAdmissionContext(
                tenant_id="tenant_a",
                governance_approved=True,
                human_approval_granted=False,
            ),
            task_id=mint_task_id(),
            production_target=True,
        )
