# © Artur Czarnecki. All rights reserved.

"""Governed sandbox side-effect helpers for DS-E2E-05."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.decision_authorization import (
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    authoritative_decision_ref,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision

from testing_support.decision_e2e.payloads import SandboxSideEffectRecord


@dataclass(slots=True)
class SandboxSideEffectStore:
    """Deterministic sandbox side-effect sink."""

    records: list[SandboxSideEffectRecord] = field(default_factory=list)

    def execute_allow(
        self,
        *,
        decision: AuthoritativeAcceptedDecision[object],
        action_kind: str,
    ) -> SandboxSideEffectRecord:
        record = SandboxSideEffectRecord(
            tenant_id=decision.identity.tenant_id,
            decision_id=str(decision.identity.decision_id),
            decision_version=decision.identity.version.value,
            action_kind=action_kind,
            executed=True,
        )
        self.records.append(record)
        return record

    def count_for_decision_version(
        self,
        *,
        tenant_id: str,
        decision_id: str,
        decision_version: str,
    ) -> int:
        return sum(
            1
            for item in self.records
            if (
                item.tenant_id == tenant_id
                and item.decision_id == decision_id
                and item.decision_version == decision_version
                and item.executed
            )
        )


@dataclass(frozen=True, slots=True)
class PolicyGovernanceEvaluator:
    """Governance evaluator with explicit ALLOW/DENY policy."""

    action: object
    policy_context: object
    allow: bool

    def evaluate(self, *, evaluation_input):
        disposition = (
            DecisionGovernanceDisposition.ALLOW
            if self.allow
            else DecisionGovernanceDisposition.DENY
        )
        return DecisionGovernanceDecision(
            disposition=disposition,
            decision_ref=authoritative_decision_ref(evaluation_input.decision),
            action=self.action,
            policy_context=self.policy_context,
            tenant_id=evaluation_input.decision.identity.tenant_id,
        )
