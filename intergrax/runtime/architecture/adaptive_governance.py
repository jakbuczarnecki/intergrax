# © Artur Czarnecki. All rights reserved.

"""Bounded adaptive loop and policy-learning governance contracts (Phase V-L4)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AdaptiveLoopKind(str, Enum):
    ROUTING_TUNING = "routing_tuning"
    EXECUTION_STRATEGY_TUNING = "execution_strategy_tuning"
    POLICY_LEARNING = "policy_learning"
    EVALUATION_FEEDBACK = "evaluation_feedback"


class AdaptiveAuthorityLevel(str, Enum):
    OBSERVE_ONLY = "observe_only"
    RECOMMEND = "recommend"
    AUTO_WITH_HUMAN_GATE = "auto_with_human_gate"


class AdaptiveLoopEnvelope(BaseModel):
    loop_id: str
    kind: AdaptiveLoopKind
    max_iterations: int = Field(ge=1, le=100)
    max_delta_percent: float = Field(ge=0.0, le=100.0)
    authority: AdaptiveAuthorityLevel
    requires_human_approval: bool
    audit_trail_required: bool = True
    cooldown_seconds: int = Field(ge=0, default=300)


class AdaptiveLoopProposal(BaseModel):
    envelope: AdaptiveLoopEnvelope
    proposed_change_summary: str
    human_approver_id: str | None = None
    evaluation_signal_id: str | None = None

    @field_validator("proposed_change_summary")
    @classmethod
    def _validate_summary(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("proposed_change_summary must not be empty")
        return normalized


class AdaptiveLoopGateResult(BaseModel):
    loop_id: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class AdaptiveGovernanceReport(BaseModel):
    schema_version: str = "1.0.0"
    proposals: list[AdaptiveLoopProposal] = Field(default_factory=list)
    results: list[AdaptiveLoopGateResult] = Field(default_factory=list)
    passed: bool


def evaluate_bounded_adaptive_loop(proposal: AdaptiveLoopProposal) -> AdaptiveLoopGateResult:
    """Validate a single adaptive loop proposal against bounded governance rules."""
    envelope = proposal.envelope
    reasons: list[str] = []

    if not envelope.audit_trail_required:
        reasons.append("Adaptive loops must require audit trail capture")

    if envelope.kind == AdaptiveLoopKind.POLICY_LEARNING:
        if not envelope.requires_human_approval:
            reasons.append("Policy learning loops require human approval")
        if not proposal.human_approver_id:
            reasons.append("Policy learning loops require human_approver_id")
        if envelope.max_delta_percent > 25.0:
            reasons.append("Policy learning delta must stay within 25 percent envelope")

    if envelope.requires_human_approval and not proposal.human_approver_id:
        reasons.append("Human approval requested but human_approver_id is missing")

    if envelope.authority == AdaptiveAuthorityLevel.OBSERVE_ONLY and envelope.requires_human_approval:
        reasons.append("Observe-only loops cannot require human approval gates")

    if envelope.authority == AdaptiveAuthorityLevel.AUTO_WITH_HUMAN_GATE:
        if not envelope.requires_human_approval:
            reasons.append("Auto-with-human-gate authority requires human approval flag")

    if envelope.max_iterations > 50 and envelope.kind != AdaptiveLoopKind.EVALUATION_FEEDBACK:
        reasons.append("Non-evaluation adaptive loops are limited to 50 iterations")

    return AdaptiveLoopGateResult(
        loop_id=envelope.loop_id,
        passed=not reasons,
        reasons=reasons,
    )


def build_default_adaptive_proposals() -> list[AdaptiveLoopProposal]:
    """Harness baseline proposals used for report generation and gate evidence."""
    return [
        AdaptiveLoopProposal(
            envelope=AdaptiveLoopEnvelope(
                loop_id="routing-tuning-01",
                kind=AdaptiveLoopKind.ROUTING_TUNING,
                max_iterations=5,
                max_delta_percent=10.0,
                authority=AdaptiveAuthorityLevel.RECOMMEND,
                requires_human_approval=False,
            ),
            proposed_change_summary="Recommend retrieval tier adjustment from evaluation drift",
            evaluation_signal_id="eval.routing.drift.001",
        ),
        AdaptiveLoopProposal(
            envelope=AdaptiveLoopEnvelope(
                loop_id="policy-learning-01",
                kind=AdaptiveLoopKind.POLICY_LEARNING,
                max_iterations=3,
                max_delta_percent=15.0,
                authority=AdaptiveAuthorityLevel.AUTO_WITH_HUMAN_GATE,
                requires_human_approval=True,
            ),
            proposed_change_summary="Tighten tool deny list using adversarial evaluation feedback",
            human_approver_id="owner:harness-security",
            evaluation_signal_id="eval.policy.adversarial.002",
        ),
        AdaptiveLoopProposal(
            envelope=AdaptiveLoopEnvelope(
                loop_id="evaluation-feedback-01",
                kind=AdaptiveLoopKind.EVALUATION_FEEDBACK,
                max_iterations=20,
                max_delta_percent=5.0,
                authority=AdaptiveAuthorityLevel.OBSERVE_ONLY,
                requires_human_approval=False,
            ),
            proposed_change_summary="Observe benchmark regression deltas without auto-apply",
            evaluation_signal_id="eval.benchmark.regression.003",
        ),
    ]


def evaluate_adaptive_governance(
    proposals: list[AdaptiveLoopProposal],
) -> AdaptiveGovernanceReport:
    results = [evaluate_bounded_adaptive_loop(proposal) for proposal in proposals]
    return AdaptiveGovernanceReport(
        proposals=proposals,
        results=results,
        passed=all(result.passed for result in results),
    )
