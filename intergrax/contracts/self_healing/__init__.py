# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing strategy contracts (SELF-HEALING R1)."""

from intergrax.contracts.self_healing.action_provider import SelfHealingActionProvider
from intergrax.contracts.self_healing.audit import SelfHealingAuditRecord
from intergrax.contracts.self_healing.context import (
    SelfHealingContext,
    SelfHealingDiagnosticInvestigation,
    SelfHealingHistoricalOutcome,
    SelfHealingOperationDescriptor,
    SelfHealingPolicyConstraints,
    SelfHealingPredictiveSignal,
)
from intergrax.contracts.self_healing.decision import (
    SelfHealingDecision,
    SelfHealingProposedAction,
    mint_self_healing_decision_id,
)
from intergrax.contracts.self_healing.governance import (
    SelfHealingAdmissionContext,
    SelfHealingAdmissionDecision,
    SelfHealingAdmissionGate,
    SelfHealingAdmissionVerdict,
)
from intergrax.contracts.self_healing.quality import SelfHealingStrategyQualityProfile
from intergrax.contracts.self_healing.registry import SelfHealingStrategyRegistry
from intergrax.contracts.self_healing.result import (
    SelfHealingExecutionOutcome,
    SelfHealingStrategyEvaluationResult,
    SelfHealingStrategyEvaluationStatus,
)
from intergrax.contracts.self_healing.safety import (
    assert_decision_has_no_execution_surface,
    assert_no_secrets_in_self_healing_audit,
    assert_strategy_has_no_execution_surface,
)
from intergrax.contracts.self_healing.strategy import (
    SelfHealingStrategy,
    SelfHealingStrategyDescriptor,
)

__all__ = [
    "SelfHealingActionProvider",
    "SelfHealingAdmissionContext",
    "SelfHealingAdmissionDecision",
    "SelfHealingAdmissionGate",
    "SelfHealingAdmissionVerdict",
    "SelfHealingAuditRecord",
    "SelfHealingContext",
    "SelfHealingDecision",
    "SelfHealingDiagnosticInvestigation",
    "SelfHealingExecutionOutcome",
    "SelfHealingHistoricalOutcome",
    "SelfHealingOperationDescriptor",
    "SelfHealingPolicyConstraints",
    "SelfHealingPredictiveSignal",
    "SelfHealingProposedAction",
    "SelfHealingStrategy",
    "SelfHealingStrategyDescriptor",
    "SelfHealingStrategyEvaluationResult",
    "SelfHealingStrategyEvaluationStatus",
    "SelfHealingStrategyQualityProfile",
    "SelfHealingStrategyRegistry",
    "assert_decision_has_no_execution_surface",
    "assert_no_secrets_in_self_healing_audit",
    "assert_strategy_has_no_execution_surface",
    "mint_self_healing_decision_id",
]
