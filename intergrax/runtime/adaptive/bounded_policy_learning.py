# © Artur Czarnecki. All rights reserved.

"""Bounded policy learning guardrails (AUDIT-IDEAL-AHI.2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.adaptation_models import AdaptationProposalPackage
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveLoopKind,
    evaluate_bounded_adaptive_loop,
)


class BoundedPolicyLearningReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires_human_approval: bool
    governance_drift_risk: str
    bounded: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_bounded_policy_learning(
    package: AdaptationProposalPackage,
) -> BoundedPolicyLearningReport:
    """Ensure policy-learning proposals remain HITL-gated and governance-bounded."""
    proposal = package.candidate.proposal
    kind = proposal.envelope.kind
    if kind != AdaptiveLoopKind.POLICY_LEARNING:
        return BoundedPolicyLearningReport(
            requires_human_approval=False,
            governance_drift_risk="none",
            bounded=True,
            reasons=["not_policy_learning"],
        )
    gate = evaluate_bounded_adaptive_loop(proposal)
    return BoundedPolicyLearningReport(
        requires_human_approval=proposal.envelope.requires_human_approval,
        governance_drift_risk="low" if gate.passed else "high",
        bounded=gate.passed,
        reasons=gate.reasons,
    )
