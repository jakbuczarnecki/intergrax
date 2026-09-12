# © Artur Czarnecki. All rights reserved.

"""ERL-QUAL-004 full execution integration — variants A/B/C via dataset-driven facts."""

from __future__ import annotations

import pytest

from intergrax.contracts.enterprise_reliability import (
    EvidenceEvaluationOutcome,
    ExternalEffectEvidenceVerdict,
    GovernanceDisposition,
    RecoveryLifecycleAction,
    ResolutionPlatformAction,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.execution import (
    EnterprisePaymentScenarioExecutionRequest,
    ScenarioLifecycleOutcome,
    build_lab_execution_composition,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningOutcomeStatus,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def executor():
    return build_lab_execution_composition()


def _run_variant(executor, variant_id: str):
    return executor.execute(
        EnterprisePaymentScenarioExecutionRequest(
            variant_id=variant_id,
            run_id=f"e2e-{variant_id}",
        ),
    )


def test_variant_a_full_execution_recovery_continuation(executor) -> None:
    result = _run_variant(executor, "payment_completed_after_unknown")
    assert result.provisioning_status is ProvisioningOutcomeStatus.SUCCEEDED
    assert result.lifecycle_outcome is ScenarioLifecycleOutcome.RECOVERY_CONTINUATION
    assert result.reconciliation_probe_verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS
    assert result.evidence_evaluation_outcome is EvidenceEvaluationOutcome.READY_FOR_DECISION
    assert result.resolution_result is not None
    assert result.resolution_result.action is ResolutionPlatformAction.CONTINUE
    assert result.recovery_result is not None
    assert result.recovery_result.action is RecoveryLifecycleAction.CONTINUE
    assert result.governance_result is not None
    assert result.governance_result.disposition is GovernanceDisposition.APPROVAL_REQUIRED
    assert result.evidence_ref is not None


def test_variant_b_full_execution_controlled_stop(executor) -> None:
    result = _run_variant(executor, "payment_failed_after_unknown")
    assert result.lifecycle_outcome is ScenarioLifecycleOutcome.CONTROLLED_STOP
    assert result.reconciliation_probe_verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE
    assert result.resolution_result is not None
    assert result.resolution_result.action is ResolutionPlatformAction.STOP
    assert result.recovery_result is not None
    assert result.recovery_result.action is RecoveryLifecycleAction.TERMINATE


def test_variant_c_full_execution_safe_escalation(executor) -> None:
    result = _run_variant(executor, "payment_truth_unavailable")
    assert result.lifecycle_outcome is ScenarioLifecycleOutcome.SAFE_ESCALATION_OR_WAIT
    assert result.reconciliation_probe_verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT
    assert result.resolution_result is not None
    assert result.resolution_result.action is ResolutionPlatformAction.ESCALATE
    assert result.recovery_result is not None
    assert result.recovery_result.action is RecoveryLifecycleAction.ESCALATE
