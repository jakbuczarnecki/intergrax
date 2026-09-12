# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing workflow contracts (SELF-HEALING R2)."""

from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext
from intergrax.contracts.self_healing.workflow.errors import (
    PLUGIN_FAILED,
    SelfHealingWorkflowError,
    SelfHealingWorkflowGovernanceError,
    SelfHealingWorkflowPluginFailedError,
    SelfHealingWorkflowStateError,
    SelfHealingWorkflowValidationError,
)
from intergrax.contracts.self_healing.workflow.lifecycle import (
    SelfHealingWorkflowAuditEntry,
    SelfHealingWorkflowState,
    assert_workflow_transition,
    mint_self_healing_workflow_id,
)
from intergrax.contracts.self_healing.workflow.outcome import SelfHealingWorkflowOutcome
from intergrax.contracts.self_healing.workflow.plan import (
    SelfHealingPlan,
    SelfHealingRiskLevel,
    mint_self_healing_plan_id,
)
from intergrax.contracts.self_healing.workflow.plan_builder import SelfHealingPlanBuilder
from intergrax.contracts.self_healing.workflow.registry import (
    SelfHealingPlanBuilderRegistry,
    SelfHealingRollbackRegistry,
    SelfHealingValidationRegistry,
    SelfHealingWorkflowPluginDescriptor,
)
from intergrax.contracts.self_healing.workflow.rollback import (
    SelfHealingRollbackDirective,
    SelfHealingRollbackPlan,
    SelfHealingRollbackProvider,
)
from intergrax.contracts.self_healing.workflow.step import SelfHealingStep
from intergrax.contracts.self_healing.workflow.validation import (
    SelfHealingValidationProvider,
    ValidationResult,
    ValidationStatus,
)

__all__ = [
    "PLUGIN_FAILED",
    "SelfHealingPlan",
    "SelfHealingPlanBuilder",
    "SelfHealingPlanBuilderRegistry",
    "SelfHealingRiskLevel",
    "SelfHealingRollbackDirective",
    "SelfHealingRollbackPlan",
    "SelfHealingRollbackProvider",
    "SelfHealingRollbackRegistry",
    "SelfHealingStep",
    "SelfHealingValidationProvider",
    "SelfHealingValidationRegistry",
    "SelfHealingWorkflowAuditEntry",
    "SelfHealingWorkflowContext",
    "SelfHealingWorkflowError",
    "SelfHealingWorkflowGovernanceError",
    "SelfHealingWorkflowOutcome",
    "SelfHealingWorkflowPluginFailedError",
    "SelfHealingWorkflowPluginDescriptor",
    "SelfHealingWorkflowState",
    "SelfHealingWorkflowStateError",
    "SelfHealingWorkflowValidationError",
    "ValidationResult",
    "ValidationStatus",
    "assert_workflow_transition",
    "mint_self_healing_plan_id",
    "mint_self_healing_workflow_id",
]
