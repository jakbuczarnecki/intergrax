# © Artur Czarnecki. All rights reserved.

"""AUTONOMOUS-ENTERPRISE-SELF-HEALING-ORCHESTRATION R3 qualification matrix."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.external_operations.admission import (
    OperationAdmissionDecision,
    OperationAdmissionVerdict,
)
from intergrax.contracts.self_healing.execution.lifecycle import SelfHealingExecutionLifecycleState
from intergrax.contracts.self_healing.governance import SelfHealingAdmissionContext
from intergrax.contracts.self_healing.selection.performance import SelfHealingStrategyPerformance
from intergrax.contracts.self_healing.strategy import SelfHealingStrategy
from intergrax.contracts.self_healing.validation.decision import ValidationDecisionStatus
from intergrax.contracts.self_healing.validation.validator import (
    ValidatorCheckResult,
    ValidatorCheckStatus,
)
from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
from intergrax.contracts.self_healing.workflow.errors import PLUGIN_FAILED, SelfHealingWorkflowPluginFailedError
from intergrax.contracts.self_healing.workflow.lifecycle import SelfHealingWorkflowState
from intergrax.contracts.self_healing.workflow.registry import SelfHealingWorkflowPluginDescriptor
from intergrax.runtime.external_operations.admission.execution_gate import ExternalOperationExecutionGate
from intergrax.runtime.external_operations.admission.local_admission import PolicyExternalOperationAdmission
from intergrax.runtime.self_healing import SelfHealingDecisionEngine
from intergrax.runtime.self_healing.defaults import platform_default_strategies
from intergrax.runtime.self_healing.lifecycle import (
    HighestConfidenceStrategySelector,
    InMemorySelfHealingStrategyPerformanceStore,
    SelfHealingLifecycleEngine,
    SelfHealingRollbackCoordinator,
    SelfHealingStrategyPerformanceEngine,
    SelfHealingValidationPipeline,
    project_healing_execution_timeline,
)
from intergrax.runtime.self_healing.workflow import (
    InMemorySelfHealingPlanBuilderRegistry,
    InMemorySelfHealingRollbackRegistry,
    InMemorySelfHealingValidationRegistry,
    SelfHealingWorkflowOrchestrator,
)
from intergrax.runtime.self_healing.workflow.bootstrap import register_platform_workflow_plugins
from tests.unit.runtime.self_healing.test_autonomous_enterprise_self_healing_strategy_r1_q import (
    _approved_context,
    _orchestrator,
    _registry_with_defaults,
    _sample_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _AllowAllExternalAdmission:
    def evaluate(self, intent: object, context: object) -> OperationAdmissionDecision:
        return OperationAdmissionDecision(verdict=OperationAdmissionVerdict.ALLOW, reason="test")


class _FailingValidator:
    validator_id = "test.failing.validator"

    def validate(
        self,
        workflow_context: SelfHealingWorkflowContext,
        *,
        execution_context: object,
        observation: object = None,
    ) -> ValidatorCheckResult:
        return ValidatorCheckResult(
            check_id="sla",
            status=ValidatorCheckStatus.FAILED,
            evidence_refs=workflow_context.evidence_refs,
            detail="sla breach",
        )


class _ExplodingValidator:
    validator_id = "test.exploding.validator"

    def validate(self, *args: object, **kwargs: object) -> ValidatorCheckResult:
        raise RuntimeError("boom")


class _CostOptimizedSelector:
    selector_id = "test.cost_optimized"

    def select(self, available_strategies: tuple[SelfHealingStrategy, ...], context: object, **kwargs: object):
        return tuple(reversed(available_strategies))


def _lifecycle_stack(
    *,
    validation_registry: InMemorySelfHealingValidationRegistry | None = None,
) -> SelfHealingLifecycleEngine:
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    decision = engine.select_and_evaluate(_sample_context())[-1].decision
    assert decision is not None
    plan_builders = InMemorySelfHealingPlanBuilderRegistry()
    val_reg = validation_registry or InMemorySelfHealingValidationRegistry()
    rollback_reg = InMemorySelfHealingRollbackRegistry()
    register_platform_workflow_plugins(
        plan_builders=plan_builders,
        validation_registry=val_reg,
        rollback_registry=rollback_reg,
        strategy_ids=(decision.strategy_id,),
    )
    healing = _orchestrator(external_admission=_AllowAllExternalAdmission())
    wf_orch = SelfHealingWorkflowOrchestrator(
        healing_orchestrator=healing,
        plan_builders=plan_builders,
        validation_registry=val_reg,
        rollback_registry=rollback_reg,
    )
    pipeline = SelfHealingValidationPipeline(validation_registry=val_reg)
    rollback = SelfHealingRollbackCoordinator(
        workflow_orchestrator=wf_orch,
        rollback_registry=rollback_reg,
    )
    return SelfHealingLifecycleEngine(
        workflow_orchestrator=wf_orch,
        validation_pipeline=pipeline,
        rollback_coordinator=rollback,
    )


def _decision():
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    decision = engine.select_and_evaluate(_sample_context())[-1].decision
    assert decision is not None
    return decision


def test_healing_workflow_full_lifecycle() -> None:
    lifecycle = _lifecycle_stack()
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    final_ctx, outcome = lifecycle.run_to_completion(
        exec_ctx.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert lifecycle.audit_trail[-1].to_state is SelfHealingExecutionLifecycleState.COMPLETED
    assert outcome.validation_result is not None
    assert final_ctx.operation_attempt_ids


def test_healing_never_executes_directly() -> None:
    lifecycle = _lifecycle_stack()
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    gate = lifecycle.workflow_orchestrator.healing_orchestrator.external_gate
    original = gate.admit_intent
    calls = {"n": 0}

    def tracked(*args: object, **kwargs: object):
        calls["n"] += 1
        return original(*args, **kwargs)

    gate.admit_intent = tracked  # type: ignore[method-assign]
    lifecycle.run_to_completion(
        exec_ctx.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert calls["n"] >= 1
    assert not hasattr(lifecycle, "external_gate")


def test_execution_id_is_preserved() -> None:
    lifecycle = _lifecycle_stack()
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    final_ctx, _ = lifecycle.run_to_completion(
        exec_ctx.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    audit_ids = {
        r.execution_id
        for r in lifecycle.workflow_orchestrator.healing_orchestrator.audit_records
        if r.execution_id
    }
    assert set(final_ctx.execution_ids).issubset(audit_ids)
    assert final_ctx.operation_attempt_ids


def test_validation_requires_observation_evidence() -> None:
    lifecycle = _lifecycle_stack()
    pipeline = lifecycle.validation_pipeline
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    wf = lifecycle.workflow_orchestrator._workflows[exec_ctx.workflow_id]  # noqa: SLF001
    with pytest.raises(ValueError, match="observation"):
        pipeline.evaluate(wf, execution_context=exec_ctx, observation=None)


def test_failed_validation_triggers_rollback() -> None:
    val_reg = InMemorySelfHealingValidationRegistry()
    lifecycle = _lifecycle_stack(validation_registry=val_reg)
    failing = _FailingValidator()
    lifecycle.validation_pipeline.register_validator(
        failing,
        SelfHealingWorkflowPluginDescriptor(
            plugin_id=failing.validator_id,
            version="1",
            namespace="test",
            priority=1,
            capabilities=("validate",),
            tenant_scope=None,
            timeout_seconds=1.0,
        ),
    )
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    _, outcome = lifecycle.run_to_completion(
        exec_ctx.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert outcome.rollback_executed
    assert lifecycle.audit_trail[-1].to_state is SelfHealingExecutionLifecycleState.ROLLED_BACK


def test_custom_validator_plugin() -> None:
    val_reg = InMemorySelfHealingValidationRegistry()
    lifecycle = _lifecycle_stack(validation_registry=val_reg)
    custom = _FailingValidator()
    lifecycle.validation_pipeline.register_validator(
        custom,
        SelfHealingWorkflowPluginDescriptor(
            plugin_id=custom.validator_id,
            version="1",
            namespace="test",
            priority=1,
            capabilities=("validate",),
            tenant_scope=None,
            timeout_seconds=1.0,
        ),
    )
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    _, outcome = lifecycle.run_to_completion(
        exec_ctx.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert outcome.validation_result is not None
    assert "sla" in outcome.validation_result.explanation or outcome.rollback_executed


def test_custom_strategy_selector() -> None:
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry, strategy_selector=_CostOptimizedSelector())
    results = engine.select_and_evaluate(_sample_context())
    assert results


def test_strategy_quality_feedback() -> None:
    lifecycle = _lifecycle_stack()
    perf_store = InMemorySelfHealingStrategyPerformanceStore()
    perf_engine = SelfHealingStrategyPerformanceEngine(perf_store)
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    _, outcome = lifecycle.run_to_completion(
        exec_ctx.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    profile = perf_engine.apply_workflow_outcome(
        outcome,
        validation_status=ValidationDecisionStatus.PASSED,
    )
    assert profile.success_rate > 0.0
    selector = HighestConfidenceStrategySelector()
    ordered = selector.select(
        platform_default_strategies(),
        _sample_context(),
        performance_profiles=(profile,),
    )
    assert ordered


def test_plugin_failure_containment() -> None:
    val_reg = InMemorySelfHealingValidationRegistry()
    lifecycle = _lifecycle_stack(validation_registry=val_reg)
    exploding = _ExplodingValidator()
    lifecycle.validation_pipeline.register_validator(
        exploding,
        SelfHealingWorkflowPluginDescriptor(
            plugin_id=exploding.validator_id,
            version="1",
            namespace="test",
            priority=1,
            capabilities=("validate",),
            tenant_scope=None,
            timeout_seconds=1.0,
        ),
    )
    decision = _decision()
    wf = lifecycle.workflow_orchestrator.create_from_decision(decision, _sample_context())
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    from intergrax.runtime.self_healing.lifecycle.default_observation import (
        PlatformExecutionObservationProvider,
    )

    obs = PlatformExecutionObservationProvider().observe(exec_ctx)
    with pytest.raises(SelfHealingWorkflowPluginFailedError) as exc:
        lifecycle.validation_pipeline.evaluate(wf, execution_context=exec_ctx, observation=obs)
    assert PLUGIN_FAILED in str(exc.value)


def test_cross_tenant_healing_isolation() -> None:
    lifecycle = _lifecycle_stack()
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    with pytest.raises(Exception):
        lifecycle.run_to_completion(
            exec_ctx.workflow_id,
            admission_context=_approved_context(tenant_id="tenant_b"),
            task_id=mint_task_id(),
        )


def test_healing_execution_timeline_projection() -> None:
    lifecycle = _lifecycle_stack()
    decision = _decision()
    exec_ctx = lifecycle.start_from_decision(decision, _sample_context())
    lifecycle.run_to_completion(
        exec_ctx.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    wf = lifecycle.workflow_orchestrator._workflows[exec_ctx.workflow_id]  # noqa: SLF001
    view = project_healing_execution_timeline(
        tuple(lifecycle.audit_trail),
        workflow_id=exec_ctx.workflow_id,
        strategy_id=decision.strategy_id,
        plan_id=wf.plan.plan_id,
    )
    assert view.stages
    assert any(s.phase == "Execution" for s in view.stages)
