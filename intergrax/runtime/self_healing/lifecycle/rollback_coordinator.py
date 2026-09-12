# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Rollback orchestration via external operation spine (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.self_healing.governance import SelfHealingAdmissionContext
from intergrax.contracts.self_healing.validation.decision import ValidationDecisionStatus
from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
from intergrax.contracts.self_healing.workflow.errors import (
    PLUGIN_FAILED,
    SelfHealingWorkflowPluginFailedError,
    SelfHealingWorkflowValidationError,
)
from intergrax.contracts.self_healing.workflow.validation import ValidationResult, ValidationStatus
from intergrax.contracts.self_healing.workflow.registry import SelfHealingRollbackRegistry
from intergrax.runtime.self_healing.workflow.orchestrator import SelfHealingWorkflowOrchestrator


@dataclass
class SelfHealingRollbackCoordinator:
    workflow_orchestrator: SelfHealingWorkflowOrchestrator
    rollback_registry: SelfHealingRollbackRegistry

    def should_rollback(self, decision_status: ValidationDecisionStatus) -> bool:
        return decision_status is ValidationDecisionStatus.FAILED

    def coordinate_rollback(
        self,
        workflow_context: SelfHealingWorkflowContext,
        *,
        admission_context: SelfHealingAdmissionContext,
        task_id: TaskId,
    ) -> SelfHealingWorkflowContext:
        provider = self.rollback_registry.resolve(
            workflow_context.plan.rollback_policy_id,
            tenant_id=workflow_context.tenant_id,
        )
        if provider is None:
            raise SelfHealingWorkflowValidationError("rollback provider not registered")
        try:
            rollback_plan = provider.plan_rollback(workflow_context)
        except Exception as exc:  # noqa: BLE001
            raise SelfHealingWorkflowPluginFailedError(
                f"{PLUGIN_FAILED}: rollback provider failed: {exc}",
            ) from exc
        if not rollback_plan.directives:
            return workflow_context
        return self.workflow_orchestrator._run_rollback(  # noqa: SLF001 — spine delegation
            workflow_context,
            admission_context=admission_context,
            task_id=task_id,
        )

    @staticmethod
    def to_legacy_validation_result(
        decision_status: ValidationDecisionStatus,
        *,
        evidence_refs: tuple[str, ...],
        confidence: float,
        explanation: str,
    ) -> ValidationResult:
        status = {
            ValidationDecisionStatus.PASSED: ValidationStatus.PASSED,
            ValidationDecisionStatus.FAILED: ValidationStatus.FAILED,
            ValidationDecisionStatus.INCONCLUSIVE: ValidationStatus.INCONCLUSIVE,
        }[decision_status]
        return ValidationResult(
            status=status,
            confidence=confidence,
            evidence_refs=evidence_refs,
            explanation=explanation,
        )


__all__ = ["SelfHealingRollbackCoordinator"]
