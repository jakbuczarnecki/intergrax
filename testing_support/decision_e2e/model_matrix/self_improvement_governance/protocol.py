# © Artur Czarnecki. All rights reserved.

"""Pluggable self-improvement governance contracts (DS-E2E-15J-L12)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    EvolutionRiskFinding,
    SelfImprovementGovernanceAuditMetadata,
    SelfImprovementGovernanceReason,
    SelfImprovementGovernanceRequest,
    SelfImprovementGovernanceStatus,
)


@dataclass(frozen=True, slots=True)
class SelfImprovementPolicyResult:
    policy_id: str
    policy_version: str
    status_contribution: SelfImprovementGovernanceStatus
    reasons: tuple[SelfImprovementGovernanceReason, ...]
    required_actions: tuple[str, ...]


class SelfImprovementPolicyEvaluator(Protocol):
    """Pluggable acceptance rule; new policy = new plugin without engine changes."""

    @property
    def policy_id(self) -> str: ...

    @property
    def policy_version(self) -> str: ...

    def evaluate(
        self, request: SelfImprovementGovernanceRequest
    ) -> SelfImprovementPolicyResult: ...


class EvolutionRiskEvaluator(Protocol):
    """Identifies risk and required controls — does not block evolution alone."""

    @property
    def evaluator_id(self) -> str: ...

    @property
    def evaluator_version(self) -> str: ...

    def assess(
        self, request: SelfImprovementGovernanceRequest
    ) -> tuple[EvolutionRiskFinding, ...]: ...


@dataclass(frozen=True, slots=True)
class SelfImprovementApprovalRecord:
    approval_id: str
    provider_id: str
    provider_version: str
    outcome_status: SelfImprovementGovernanceStatus
    rationale: str


class SelfImprovementApprovalProvider(Protocol):
    """Human, enterprise workflow, or test double — never auto-approves from confidence."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def decide(
        self,
        request: SelfImprovementGovernanceRequest,
        *,
        policy_results: tuple[SelfImprovementPolicyResult, ...],
        aggregated_status: SelfImprovementGovernanceStatus,
    ) -> SelfImprovementApprovalRecord: ...


class SelfImprovementGovernanceAuditProvider(Protocol):
    """Builds auditable metadata for every governance decision."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def build_audit(
        self,
        request: SelfImprovementGovernanceRequest,
        *,
        risk_findings: tuple[EvolutionRiskFinding, ...],
        policy_results: tuple[SelfImprovementPolicyResult, ...],
        approval_record: SelfImprovementApprovalRecord | None,
        final_status: SelfImprovementGovernanceStatus,
        evaluated_at: datetime,
        policy_evaluator_ids: tuple[str, ...],
        policy_evaluator_versions: tuple[str, ...],
        risk_evaluator_ids: tuple[str, ...],
        risk_evaluator_versions: tuple[str, ...],
    ) -> SelfImprovementGovernanceAuditMetadata: ...


__all__ = [
    "EvolutionRiskEvaluator",
    "SelfImprovementApprovalProvider",
    "SelfImprovementApprovalRecord",
    "SelfImprovementGovernanceAuditProvider",
    "SelfImprovementPolicyEvaluator",
    "SelfImprovementPolicyResult",
]
