# © Artur Czarnecki. All rights reserved.

"""AUTONOMOUS-ENTERPRISE-SELF-HEALING-ORCHESTRATION R2 qualification matrix."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.external_operations.admission import (
    OperationAdmissionDecision,
    OperationAdmissionVerdict,
)
from intergrax.contracts.self_healing.governance import SelfHealingAdmissionContext
from intergrax.contracts.self_healing.workflow.errors import (
    PLUGIN_FAILED,
    SelfHealingWorkflowGovernanceError,
    SelfHealingWorkflowPluginFailedError,
    SelfHealingWorkflowStateError,
)
from intergrax.contracts.self_healing.workflow.lifecycle import SelfHealingWorkflowState
from intergrax.contracts.self_healing.workflow.validation import ValidationResult, ValidationStatus
from intergrax.runtime.external_operations.admission.execution_gate import ExternalOperationExecutionGate
from intergrax.runtime.external_operations.admission.local_admission import (
    PolicyExternalOperationAdmission,
)
from intergrax.runtime.self_healing import (
    GovernedSelfHealingOrchestrator,
    InMemorySelfHealingStrategyRegistry,
    PlatformSelfHealingActionProvider,
    PolicySelfHealingAdmissionGate,
    SelfHealingDecisionEngine,
    platform_default_strategies,
)
from intergrax.runtime.self_healing.outcome_learning import InMemorySelfHealingStrategyQualityStore
from intergrax.runtime.self_healing.workflow import (
    InMemorySelfHealingPlanBuilderRegistry,
    InMemorySelfHealingRollbackRegistry,
    InMemorySelfHealingValidationRegistry,
    SelfHealingWorkflowOrchestrator,
    SelfHealingWorkflowOutcomeEngine,
    project_healing_workflow_history,
)
from intergrax.runtime.self_healing.workflow.bootstrap import register_platform_workflow_plugins
from intergrax.contracts.self_healing.workflow.registry import SelfHealingWorkflowPluginDescriptor
from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
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


class _FailingValidation:
    provider_id = "platform.default.validation"

    def validate(self, workflow_context: SelfHealingWorkflowContext) -> ValidationResult:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            confidence=0.1,
            evidence_refs=workflow_context.evidence_refs,
            explanation="forced failure for test",
        )


class _ExplodingValidation:
    provider_id = "platform.default.validation"

    def validate(self, workflow_context: SelfHealingWorkflowContext) -> ValidationResult:
        raise RuntimeError("boom")


def _workflow_stack(
    *,
    validation_registry: InMemorySelfHealingValidationRegistry | None = None,
    register_platform_validation: bool = True,
) -> SelfHealingWorkflowOrchestrator:
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
        register_validation=register_platform_validation,
    )
    healing = _orchestrator(external_admission=_AllowAllExternalAdmission())
    return SelfHealingWorkflowOrchestrator(
        healing_orchestrator=healing,
        plan_builders=plan_builders,
        validation_registry=val_reg,
        rollback_registry=rollback_reg,
    )


def _decision():
    registry = _registry_with_defaults()
    engine = SelfHealingDecisionEngine(registry)
    decision = engine.select_and_evaluate(_sample_context())[-1].decision
    assert decision is not None
    return decision


def test_strategy_creates_healing_plan() -> None:
    wf_orch = _workflow_stack()
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    assert wf.plan.plan_id.startswith("sh_plan_")
    assert len(wf.plan.steps) >= 2
    assert wf.state is SelfHealingWorkflowState.PLANNED


def test_multi_step_workflow_execution() -> None:
    wf_orch = _workflow_stack()
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    final, outcome = wf_orch.run_to_completion(
        wf.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert final.state is SelfHealingWorkflowState.SUCCEEDED
    assert len(outcome.successful_steps) >= 2


def test_healing_uses_external_operation_spine() -> None:
    wf_orch = _workflow_stack()
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    wf_orch.run_to_completion(
        wf.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert wf_orch.healing_orchestrator.audit_records
    assert wf_orch.healing_orchestrator.audit_records[0].external_operation_id


def test_healing_cannot_bypass_governance() -> None:
    wf_orch = _workflow_stack()
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    with pytest.raises(SelfHealingWorkflowGovernanceError):
        wf_orch.run_to_completion(
            wf.workflow_id,
            admission_context=SelfHealingAdmissionContext(
                tenant_id="tenant_a",
                governance_approved=False,
            ),
            task_id=mint_task_id(),
        )


def test_success_requires_validation() -> None:
    wf_orch = _workflow_stack()
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    final, _ = wf_orch.run_to_completion(
        wf.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert final.validation_result is not None
    assert final.validation_result.status is ValidationStatus.PASSED


def test_failed_validation_requires_rollback() -> None:
    val_reg = InMemorySelfHealingValidationRegistry()
    failing = _FailingValidation()
    val_reg.register(
        failing,
        SelfHealingWorkflowPluginDescriptor(
            plugin_id=failing.provider_id,
            version="1",
            namespace="test",
            priority=1,
            capabilities=("validate",),
            tenant_scope=None,
            timeout_seconds=1.0,
        ),
    )
    wf_orch = _workflow_stack(
        validation_registry=val_reg,
        register_platform_validation=False,
    )
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    final, outcome = wf_orch.run_to_completion(
        wf.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert final.state is SelfHealingWorkflowState.FAILED
    assert outcome.rollback_executed


def test_custom_validation_plugin() -> None:
    val_reg = InMemorySelfHealingValidationRegistry()
    custom = _FailingValidation()
    val_reg.register(
        custom,
        SelfHealingWorkflowPluginDescriptor(
            plugin_id=custom.provider_id,
            version="1",
            namespace="test",
            priority=1,
            capabilities=("validate",),
            tenant_scope=None,
            timeout_seconds=1.0,
        ),
    )
    wf_orch = _workflow_stack(
        validation_registry=val_reg,
        register_platform_validation=False,
    )
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    final, outcome = wf_orch.run_to_completion(
        wf.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    assert outcome.validation_result is not None
    assert outcome.validation_result.explanation == "forced failure for test"


def test_failed_plugin_is_contained() -> None:
    val_reg = InMemorySelfHealingValidationRegistry()
    exploding = _ExplodingValidation()
    val_reg.register(
        exploding,
        SelfHealingWorkflowPluginDescriptor(
            plugin_id=exploding.provider_id,
            version="1",
            namespace="test",
            priority=1,
            capabilities=("validate",),
            tenant_scope=None,
            timeout_seconds=1.0,
        ),
    )
    wf_orch = _workflow_stack(
        validation_registry=val_reg,
        register_platform_validation=False,
    )
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    wf_orch._transition(wf, SelfHealingWorkflowState.APPROVED, actor="test", reason="test")  # noqa: SLF001
    wf_orch._workflows[wf.workflow_id] = wf  # noqa: SLF001
    with pytest.raises(SelfHealingWorkflowPluginFailedError) as exc:
        wf_orch._run_validation(wf)  # noqa: SLF001
    assert PLUGIN_FAILED in str(exc.value)


def test_strategy_quality_updates_from_outcome() -> None:
    store = InMemorySelfHealingStrategyQualityStore()
    engine = SelfHealingWorkflowOutcomeEngine(store)
    wf_orch = _workflow_stack()
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    _, outcome = wf_orch.run_to_completion(
        wf.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    profile = engine.apply_workflow_outcome(outcome)
    assert profile.successful_preventions >= 1


def test_healing_workflow_tenant_isolation() -> None:
    wf_orch = _workflow_stack()
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    with pytest.raises(SelfHealingWorkflowStateError):
        wf_orch.run_to_completion(
            wf.workflow_id,
            admission_context=_approved_context(tenant_id="tenant_b"),
            task_id=mint_task_id(),
        )


def test_workflow_history_projects_to_read_model() -> None:
    wf_orch = _workflow_stack()
    decision = _decision()
    wf = wf_orch.create_from_decision(decision, _sample_context())
    wf_orch.run_to_completion(
        wf.workflow_id,
        admission_context=_approved_context(),
        task_id=mint_task_id(),
    )
    views = project_healing_workflow_history(
        tuple(wf_orch.audit_trail),
        workflow_id=wf.workflow_id,
        strategy_id=decision.strategy_id,
        plan_id=wf.plan.plan_id,
    )
    assert views
    assert views[-1].to_state == SelfHealingWorkflowState.SUCCEEDED.value
