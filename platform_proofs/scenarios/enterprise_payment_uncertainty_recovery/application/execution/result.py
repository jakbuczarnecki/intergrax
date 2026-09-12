"""Proof-oriented execution summary — references platform contracts, no duplicates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.enterprise_reliability.evidence_evaluation import EvidenceEvaluationOutcome
from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.governance_decision import GovernanceDecision
from intergrax.contracts.enterprise_reliability.recovery_decision import (
    RecoveryDecision,
    RecoveryLifecycleAction,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.domain.workflow import (
    BusinessWorkflowPhase,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
)


class ScenarioLifecycleOutcome(StrEnum):
    """Terminal posture derived from recovery and governance facts."""

    RECOVERY_CONTINUATION = "recovery_continuation"
    CONTROLLED_STOP = "controlled_stop"
    SAFE_ESCALATION_OR_WAIT = "safe_escalation_or_wait"
    PROVISIONING_FAILED = "provisioning_failed"
    APPLICATION_INCOMPLETE = "application_incomplete"


@dataclass(frozen=True, slots=True)
class ScenarioExecutionProofResult:
    scenario_id: str
    variant_id: str
    correlation_id: str
    lifecycle_outcome: ScenarioLifecycleOutcome
    provisioning_status: ProvisioningOutcomeStatus
    application_phase: BusinessWorkflowPhase | None
    evidence_ref: str | None
    evidence_evaluation_outcome: EvidenceEvaluationOutcome | None
    reconciliation_probe_verdict: ExternalEffectEvidenceVerdict | None
    resolution_result: ResolutionDecision | None
    governance_result: GovernanceDecision | None
    recovery_result: RecoveryDecision | None


def derive_lifecycle_outcome(
    *,
    provisioning_status: ProvisioningOutcomeStatus,
    application_phase: BusinessWorkflowPhase | None,
    recovery: RecoveryDecision | None,
) -> ScenarioLifecycleOutcome:
    if provisioning_status is not ProvisioningOutcomeStatus.SUCCEEDED:
        return ScenarioLifecycleOutcome.PROVISIONING_FAILED
    if application_phase is not BusinessWorkflowPhase.PAYMENT_REQUESTED:
        return ScenarioLifecycleOutcome.APPLICATION_INCOMPLETE
    if recovery is None:
        return ScenarioLifecycleOutcome.APPLICATION_INCOMPLETE
    if recovery.action is RecoveryLifecycleAction.CONTINUE:
        return ScenarioLifecycleOutcome.RECOVERY_CONTINUATION
    if recovery.action is RecoveryLifecycleAction.TERMINATE:
        return ScenarioLifecycleOutcome.CONTROLLED_STOP
    if recovery.action in (
        RecoveryLifecycleAction.ESCALATE,
        RecoveryLifecycleAction.WAIT,
    ):
        return ScenarioLifecycleOutcome.SAFE_ESCALATION_OR_WAIT
    return ScenarioLifecycleOutcome.APPLICATION_INCOMPLETE
