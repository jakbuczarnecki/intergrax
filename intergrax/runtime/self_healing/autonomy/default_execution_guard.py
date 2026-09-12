# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default autonomy execution guard — check only, never execute (SELF-HEALING R6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.contracts.self_healing.autonomy.evaluation_repository import AutonomyDecisionRepository
from intergrax.contracts.self_healing.autonomy.execution_audit import (
    AutonomyExecutionAuditRecord,
    AutonomyExecutionAuditRepository,
)
from intergrax.contracts.self_healing.autonomy.execution_authorization import AutonomyExecutionAuthorization
from intergrax.contracts.self_healing.autonomy.guard import (
    AutonomyExecutionAdmissionContext,
    AutonomyGuardCheckResult,
)
from intergrax.contracts.self_healing.autonomy.guard_rule import AutonomyExecutionGuardRule
from intergrax.contracts.self_healing.autonomy.ids import mint_autonomy_execution_audit_record_id
from intergrax.contracts.self_healing.autonomy.repository import AutonomyDecisionCorrelationQuery
from intergrax.runtime.self_healing.autonomy.guard_support import (
    authorization_to_guard_verdict,
    resolve_execution_authorization,
)


@dataclass(frozen=True, slots=True)
class DefaultAutonomyExecutionGuard:
    evaluation_repository: AutonomyDecisionRepository
    audit_repository: AutonomyExecutionAuditRepository | None = None
    guard_rules: tuple[AutonomyExecutionGuardRule, ...] = ()
    _guard_id: str = "platform.default_autonomy_execution_guard"

    @property
    def guard_id(self) -> str:
        return self._guard_id

    def authorize(
        self,
        admission: AutonomyExecutionAdmissionContext,
    ) -> AutonomyExecutionAuthorization:
        query = AutonomyDecisionCorrelationQuery(
            tenant_id=admission.tenant_id,
            recommendation_correlation_id=admission.recommendation_correlation_id,
        )
        evaluation = self.evaluation_repository.get_latest_evaluation(query)
        recorded_at = datetime.now(tz=timezone.utc)
        authorization = resolve_execution_authorization(
            admission,
            evaluation,
            guard_id=self.guard_id,
            recorded_at=recorded_at,
            guard_rules=self.guard_rules,
        )
        if self.audit_repository is not None:
            record = AutonomyExecutionAuditRecord.from_authorization(
                mint_autonomy_execution_audit_record_id(),
                authorization,
            )
            self.audit_repository.append(record)
        return authorization

    def check(self, admission: AutonomyExecutionAdmissionContext) -> AutonomyGuardCheckResult:
        authorization = self.authorize(admission)
        verdict, rationale = authorization_to_guard_verdict(authorization)
        audit_refs: tuple[str, ...] = (authorization.authorization_id,)
        if authorization.evaluation_id is not None:
            audit_refs = (authorization.authorization_id, authorization.evaluation_id)
        return AutonomyGuardCheckResult(
            verdict=verdict,
            rationale=rationale,
            audit_refs=audit_refs,
        )


__all__ = ["DefaultAutonomyExecutionGuard"]
