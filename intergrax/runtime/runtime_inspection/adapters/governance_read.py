# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Governance audit read adapters for runtime inspection."""

from __future__ import annotations

from intergrax.contracts.agent_runtime_governance import (
    GovernanceAuditEvent,
    ToolAuthorizationDecisionState,
)
from intergrax.contracts.governance_audit_read import (
    GovernanceAuditReadIntegrityError,
    GovernanceAuditReadPort,
    GovernanceAuditReadTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
    RuntimeInspectionTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.limits import (
    DEFAULT_RUNTIME_INSPECTION_GOVERNANCE_DECISION_LIMIT,
)
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionGovernanceDecisionEntry,
    RuntimeInspectionGovernanceSection,
)
from intergrax.contracts.runtime_inspection.sources import (
    RuntimeInspectionExecutionScope,
    RuntimeInspectionGovernanceReadPort,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def _validate_governance_event_scope(
    scope: RuntimeInspectionExecutionScope,
    event: GovernanceAuditEvent,
    *,
    source_id: str,
) -> None:
    if event.tenant_id != scope.tenant_id:
        raise RuntimeInspectionTenantBoundaryError(execution_id=scope.execution_id)
    if event.task_id != scope.task_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "governance audit task_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if event.run_id != scope.run_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "governance audit run_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if event.attempt_id != scope.attempt_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "governance audit attempt_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if event.execution_id is None or event.execution_id != scope.execution_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "governance audit execution_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )


class GovernanceAuditInspectionAdapter(RuntimeInspectionGovernanceReadPort):
    def __init__(
        self,
        audit_reader: GovernanceAuditReadPort,
        *,
        decision_limit: int = DEFAULT_RUNTIME_INSPECTION_GOVERNANCE_DECISION_LIMIT,
    ) -> None:
        self._audit_reader = audit_reader
        self._decision_limit = decision_limit

    @property
    def source_id(self) -> str:
        return self._audit_reader.source_id

    def read_governance_decisions(
        self,
        scope: RuntimeInspectionExecutionScope,
    ) -> RuntimeInspectionGovernanceSection:
        try:
            events = self._audit_reader.list_audit_events_for_execution(
                tenant_id=scope.tenant_id,
                execution_id=scope.execution_id,
                limit=self._decision_limit,
            )
        except GovernanceAuditReadTenantBoundaryError as exc:
            raise RuntimeInspectionTenantBoundaryError(
                execution_id=scope.execution_id,
            ) from exc
        except GovernanceAuditReadIntegrityError as exc:
            raise RuntimeInspectionError(
                RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
                "governance audit read integrity failure",
                execution_id=scope.execution_id,
                source_id=self.source_id,
            ) from exc
        decisions: list[RuntimeInspectionGovernanceDecisionEntry] = []
        for event in events:
            _validate_governance_event_scope(
                scope,
                event,
                source_id=self.source_id,
            )
            policy_id = None
            policy_revision = None
            if event.policy_results:
                policy_id = event.policy_results[0].policy_id
            reason = sanitize_inspection_text(
                event.policy_results[0].reason if event.policy_results else event.tool_id,
            )
            decisions.append(
                RuntimeInspectionGovernanceDecisionEntry(
                    decision_ref=event.event_id,
                    outcome=event.decision,
                    policy_id=policy_id,
                    policy_revision=policy_revision,
                    reason_classification=reason,
                    approval_required=event.decision
                    is ToolAuthorizationDecisionState.REQUIRE_APPROVAL,
                    grant_ref=None,
                    evidence_refs=(event.event_id,),
                ),
            )
        completeness = (
            RuntimeInspectionCompleteness.COMPLETE
            if decisions or len(events) < self._decision_limit
            else RuntimeInspectionCompleteness.PARTIAL
        )
        return RuntimeInspectionGovernanceSection(
            decisions=tuple(decisions),
            completeness=completeness,
            source_id=self.source_id,
            source_available=True,
        )


__all__ = ["GovernanceAuditInspectionAdapter"]
