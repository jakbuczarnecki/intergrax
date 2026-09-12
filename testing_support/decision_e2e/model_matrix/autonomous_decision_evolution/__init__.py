# © Artur Czarnecki. All rights reserved.

"""Autonomous decision evolution — controlled proposals only (DS-E2E-15J-L11)."""

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.approval_providers import (
    HumanReviewApprovalProvider,
    default_approval_providers,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    AUTONOMOUS_EVOLUTION_TASK_ID,
    AUTONOMOUS_EVOLUTION_VERSION,
    AutonomousDecisionEvolutionAuditMetadata,
    AutonomousDecisionEvolutionInput,
    AutonomousDecisionEvolutionResult,
    ControlledEvolutionRecord,
    DecisionEvolutionProposal,
    EvaluationCriterion,
    EvaluationCriterionKind,
    EvolutionApprovalDecision,
    EvolutionApprovalOutcome,
    EvolutionEvaluationFinding,
    EvolutionExperimentSpec,
    EvolutionRiskInformation,
    EvolutionRunStatus,
    EvolutionTargetArea,
    ExperimentVariant,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.engine import (
    AutonomousDecisionEvolutionEngine,
    default_autonomous_decision_evolution_engine,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.evaluation_providers import (
    CostEvolutionEvaluator,
    QualityEvolutionEvaluator,
    ReliabilityEvolutionEvaluator,
    SafetyEvolutionEvaluator,
    default_evaluation_providers,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.experiment_providers import (
    StandardControlledExperimentProvider,
    default_experiment_providers,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.proposal_generators import (
    ModelChangeProposalGenerator,
    PolicyChangeProposalGenerator,
    ProcessChangeProposalGenerator,
    default_proposal_generators,
)
from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.protocol import (
    EvolutionApprovalProvider,
    EvolutionEvaluationProvider,
    EvolutionExperimentProvider,
    EvolutionProposalGenerator,
)

__all__ = [
    "AUTONOMOUS_EVOLUTION_TASK_ID",
    "AUTONOMOUS_EVOLUTION_VERSION",
    "AutonomousDecisionEvolutionAuditMetadata",
    "AutonomousDecisionEvolutionEngine",
    "AutonomousDecisionEvolutionInput",
    "AutonomousDecisionEvolutionResult",
    "ControlledEvolutionRecord",
    "CostEvolutionEvaluator",
    "DecisionEvolutionProposal",
    "EvaluationCriterion",
    "EvaluationCriterionKind",
    "EvolutionApprovalDecision",
    "EvolutionApprovalOutcome",
    "EvolutionApprovalProvider",
    "EvolutionEvaluationFinding",
    "EvolutionEvaluationProvider",
    "EvolutionExperimentProvider",
    "EvolutionExperimentSpec",
    "EvolutionProposalGenerator",
    "EvolutionRiskInformation",
    "EvolutionRunStatus",
    "EvolutionTargetArea",
    "ExperimentVariant",
    "HumanReviewApprovalProvider",
    "ModelChangeProposalGenerator",
    "PolicyChangeProposalGenerator",
    "ProcessChangeProposalGenerator",
    "QualityEvolutionEvaluator",
    "ReliabilityEvolutionEvaluator",
    "SafetyEvolutionEvaluator",
    "StandardControlledExperimentProvider",
    "default_approval_providers",
    "default_autonomous_decision_evolution_engine",
    "default_evaluation_providers",
    "default_experiment_providers",
    "default_proposal_generators",
]
