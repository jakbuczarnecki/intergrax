# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LLM proposal → Decision → Governance → Admission (R1)."""

from __future__ import annotations

from intergrax.contracts.external_operations.admission import (
    ExternalOperationAdmission,
    ExternalOperationAdmissionContext,
    OperationAdmissionVerdict,
)
from intergrax.contracts.external_operations.governance import (
    ExternalOperationGovernanceContext,
    ExternalOperationGovernanceDecision,
)
from intergrax.contracts.external_operations.intent import ExternalOperationIntent


def resolve_governance_admission(
    *,
    intent: ExternalOperationIntent,
    governance: ExternalOperationGovernanceContext,
    admission: ExternalOperationAdmission,
    provider_id: str | None = None,
    production_target: bool = False,
) -> ExternalOperationGovernanceDecision:
    """Never maps LLM output directly to execution."""
    if intent.tenant_id != governance.tenant_id:
        from intergrax.contracts.external_operations.admission import (
            OperationAdmissionDecision,
        )

        denied = OperationAdmissionDecision(
            verdict=OperationAdmissionVerdict.DENY,
            reason="governance tenant mismatch",
            decision_id=governance.decision_id,
        )
        return ExternalOperationGovernanceDecision(
            intent=intent,
            admission=denied,
            may_execute=False,
        )
    if not governance.governance_approved:
        from intergrax.contracts.external_operations.admission import (
            OperationAdmissionDecision,
        )

        denied = OperationAdmissionDecision(
            verdict=OperationAdmissionVerdict.DENY,
            reason="decision governance rejected proposal",
            decision_id=governance.decision_id,
        )
        return ExternalOperationGovernanceDecision(
            intent=intent,
            admission=denied,
            may_execute=False,
        )
    decision = admission.evaluate(
        intent,
        ExternalOperationAdmissionContext(
            tenant_id=governance.tenant_id,
            provider_id=provider_id,
            production_target=production_target,
            decision_id=governance.decision_id,
            human_approval_granted=governance.governance_approved,
            approval_id=None,
        ),
    )
    may_execute = decision.verdict is OperationAdmissionVerdict.ALLOW
    return ExternalOperationGovernanceDecision(
        intent=intent,
        admission=decision,
        may_execute=may_execute,
    )
