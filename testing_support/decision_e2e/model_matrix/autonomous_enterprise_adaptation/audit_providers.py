# © Artur Czarnecki. All rights reserved.

"""Adaptation audit metadata providers (DS-E2E-15J-L13)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from testing_support.decision_e2e.model_matrix.autonomous_enterprise_adaptation.contracts import (
    AUTONOMOUS_ENTERPRISE_ADAPTATION_TASK_ID,
    AUTONOMOUS_ENTERPRISE_ADAPTATION_VERSION,
    AdaptationAuditMetadata,
    AdaptationExecutionStatus,
    ApprovedAdaptationRequest,
)


_STANDARD_ADAPTATION_AUDIT_PROVIDER_ID = "standard_adaptation_audit"
_STANDARD_ADAPTATION_AUDIT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class StandardAdaptationAuditProvider:
    @property
    def provider_id(self) -> str:
        return _STANDARD_ADAPTATION_AUDIT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _STANDARD_ADAPTATION_AUDIT_PROVIDER_VERSION

    def build_audit(
        self,
        approved_change: ApprovedAdaptationRequest,
        *,
        outcome_status: AdaptationExecutionStatus,
        adaptation_provider_id: str,
        adaptation_provider_version: str,
        applied_change_reference: str | None,
        executed_at: datetime,
        outcome_summary: str,
    ) -> AdaptationAuditMetadata:
        approval = approved_change.governance_approval
        return AdaptationAuditMetadata(
            adaptation_task_id=AUTONOMOUS_ENTERPRISE_ADAPTATION_TASK_ID,
            adaptation_layer_version=AUTONOMOUS_ENTERPRISE_ADAPTATION_VERSION,
            adaptation_id=approved_change.adaptation_id,
            adaptation_version=approved_change.version,
            source_reference=approved_change.source_reference,
            governance_approval_id=approval.approval_id
            if approval is not None
            else None,
            approver_identity=approval.approver_identity
            if approval is not None
            else None,
            provider_id=adaptation_provider_id,
            provider_version=adaptation_provider_version,
            applied_change_reference=applied_change_reference,
            outcome_status=outcome_status,
            executed_at=executed_at,
            outcome_summary=outcome_summary,
        )


def default_adaptation_audit_provider() -> StandardAdaptationAuditProvider:
    return StandardAdaptationAuditProvider()


__all__ = [
    "StandardAdaptationAuditProvider",
    "default_adaptation_audit_provider",
]
