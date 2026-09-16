# © Artur Czarnecki. All rights reserved.

"""Memory security & governance orchestration (MEM-ENT-10)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOperation,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
    MemoryRetentionAction,
    MemorySecurityStrategySet,
)
from intergrax.memory.strategies.defaults.memory_security_governance import (
    build_default_memory_security_strategy_set,
)
from intergrax.memory.strategies.recall_models import MemoryRecallCandidate

__all__ = [
    "MemorySecurityGovernanceService",
    "build_default_memory_security_governance_service",
]


_OUTCOME_STRICTNESS: dict[MemoryGovernanceOutcome, int] = {
    MemoryGovernanceOutcome.ALLOW: 0,
    MemoryGovernanceOutcome.ALLOW_WITH_CONSTRAINTS: 1,
    MemoryGovernanceOutcome.REQUIRE_REVIEW: 2,
    MemoryGovernanceOutcome.DENY: 3,
}

_MUTATION_OPERATIONS = frozenset(
    {
        MemoryGovernanceOperation.REMEMBER,
        MemoryGovernanceOperation.PROMOTE,
        MemoryGovernanceOperation.SUPERSEDE,
        MemoryGovernanceOperation.DELETE,
        MemoryGovernanceOperation.COMPACT,
        MemoryGovernanceOperation.PROJECT,
        MemoryGovernanceOperation.UPDATE,
    }
)


def _fail_closed_decision(
    request: MemoryGovernanceEvaluationRequest,
    *,
    reason_code: MemoryGovernanceReasonCode,
    policy_id: str = "memory.security.fail_closed",
) -> MemoryGovernanceDecision:
    return MemoryGovernanceDecision(
        outcome=MemoryGovernanceOutcome.DENY,
        reason_code=reason_code,
        policy_id=policy_id,
        policy_version="1.0.0",
        operation=request.context.operation,
        subject_memory_id=(
            request.proposed_record.memory_id
            if request.proposed_record is not None
            else request.target.memory_id if request.target is not None else None
        ),
    )


def _merge_decisions(
    request: MemoryGovernanceEvaluationRequest,
    decisions: tuple[MemoryGovernanceDecision, ...],
) -> MemoryGovernanceDecision:
    if not decisions:
        return _fail_closed_decision(request, reason_code=MemoryGovernanceReasonCode.POLICY_MISSING)
    strictest = max(decisions, key=lambda d: _OUTCOME_STRICTNESS[d.outcome])
    constraints: list = []
    for decision in decisions:
        constraints.extend(decision.constraints)
    return MemoryGovernanceDecision(
        outcome=strictest.outcome,
        reason_code=strictest.reason_code,
        policy_id=strictest.policy_id,
        policy_version=strictest.policy_version,
        operation=request.context.operation,
        trust_class=strictest.trust_class,
        data_classification=strictest.data_classification,
        retention_action=strictest.retention_action,
        constraints=tuple(constraints),
        subject_memory_id=strictest.subject_memory_id,
    )


@dataclass(slots=True)
class MemorySecurityGovernanceService:
    strategies: MemorySecurityStrategySet

    def evaluate(self, request: MemoryGovernanceEvaluationRequest) -> MemoryGovernanceDecision:
        if self.strategies is None:
            return _fail_closed_decision(request, reason_code=MemoryGovernanceReasonCode.POLICY_MISSING)
        try:
            auth = self.strategies.authorization.evaluate(request)
            trust = self.strategies.trust.evaluate(request)
            admission = self.strategies.admission.evaluate(request)
            governance = self.strategies.governance.evaluate(request)
            retention = self.strategies.retention.evaluate(request)
        except Exception:
            return _fail_closed_decision(
                request,
                reason_code=MemoryGovernanceReasonCode.POLICY_FAILURE,
            )

        decisions: list[MemoryGovernanceDecision] = [auth, admission, governance]
        if trust.escalation_blocked and request.context.operation in _MUTATION_OPERATIONS:
            decisions.append(
                MemoryGovernanceDecision(
                    outcome=MemoryGovernanceOutcome.DENY,
                    reason_code=trust.reason_code,
                    policy_id=trust.policy_id,
                    policy_version=trust.policy_version,
                    operation=request.context.operation,
                    trust_class=trust.effective_trust_class,
                    subject_memory_id=(
                        request.proposed_record.memory_id
                        if request.proposed_record is not None
                        else None
                    ),
                )
            )
        if retention.action is MemoryRetentionAction.DELETE and request.context.operation in _MUTATION_OPERATIONS:
            decisions.append(
                MemoryGovernanceDecision(
                    outcome=MemoryGovernanceOutcome.DENY,
                    reason_code=MemoryGovernanceReasonCode.RETENTION_BLOCK,
                    policy_id=retention.policy_id,
                    policy_version=retention.policy_version,
                    operation=request.context.operation,
                )
            )
        merged = _merge_decisions(request, tuple(decisions))
        if not _is_valid_decision(merged):
            return _fail_closed_decision(
                request,
                reason_code=MemoryGovernanceReasonCode.POLICY_FAILURE,
            )
        return merged

    def filter_recall_candidates(
        self,
        request: MemoryGovernanceEvaluationRequest,
        candidates: tuple[MemoryRecallCandidate, ...],
    ) -> tuple[MemoryRecallCandidate, ...]:
        allowed: list[MemoryRecallCandidate] = []
        for candidate in candidates:
            from intergrax.memory.contracts.memory_security_governance import (
                MemoryGovernanceRecordSnapshot,
                MemoryGovernanceTarget,
            )

            snapshot = MemoryGovernanceRecordSnapshot.from_user_profile_entry(candidate.record)
            per_record = MemoryGovernanceEvaluationRequest(
                context=request.context,
                target=MemoryGovernanceTarget(
                    memory_id=snapshot.memory_id,
                    revision=snapshot.revision,
                    kind=snapshot.kind,
                    scope=request.context.scope,
                ),
                existing_record=snapshot,
            )
            decision = self.evaluate(per_record)
            if decision.permits_disclosure():
                allowed.append(candidate)
        return tuple(allowed)


def _is_valid_decision(decision: MemoryGovernanceDecision) -> bool:
    return (
        decision.policy_id.strip() != ""
        and decision.policy_version.strip() != ""
        and decision.reason_code is not None
        and decision.outcome is not None
    )


def build_default_memory_security_governance_service(
    *,
    strategies: MemorySecurityStrategySet | None = None,
) -> MemorySecurityGovernanceService:
    return MemorySecurityGovernanceService(
        strategies=strategies or build_default_memory_security_strategy_set(),
    )
