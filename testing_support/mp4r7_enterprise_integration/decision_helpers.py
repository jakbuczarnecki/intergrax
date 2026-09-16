# © Artur Czarnecki. All rights reserved.

"""Decision-flow qualification helpers for MP-4R7 (no production authority)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_authorization import (
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    authoritative_decision_ref,
    decision_execution_action,
    decision_governance_policy_context,
    validate_decision_execution_action_kind,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewOutcome,
    DecisionHumanReviewPending,
)
from intergrax.contracts.decision_verification import (
    VerificationStageOutcome,
    validate_verification_stage_kind,
    verification_stage_record,
)
from intergrax.contracts.decision_verification_stage import (
    VerificationStage,
    VerificationStageExecutionClass,
    VerificationStageRegistration,
    verification_stage_registry,
)
from intergrax.runtime.decision_flow import DecisionFlowGovernanceSpec
from intergrax.runtime.decision_human_review import request_decision_human_review
from intergrax.runtime.decision_verification import VerificationPipeline


@dataclass(frozen=True, slots=True)
class Mp4R7PassedVerificationStage:
    kind: str = "mp4r7.verification.stage"
    execution_class: VerificationStageExecutionClass = VerificationStageExecutionClass.DETERMINISTIC

    async def verify(self, candidate):
        from intergrax.contracts.decision_record import candidate_decision_ref

        return verification_stage_record(
            proposal_ref=candidate_decision_ref(candidate),
            stage=validate_verification_stage_kind(self.kind),
            outcome=VerificationStageOutcome.PASSED,
        )


@dataclass(slots=True)
class Mp4R7RecordingHumanReviewPort:
    pending: DecisionHumanReviewPending | None = None

    def request_review(self, request):
        self.pending = request_decision_human_review(request)
        return self.pending


@dataclass(frozen=True, slots=True)
class Mp4R7RequireHumanGovernanceEvaluator:
    action: object
    policy_context: object

    def evaluate(self, *, evaluation_input):
        human = evaluation_input.human_review_decision
        if human is not None and human.outcome is DecisionHumanReviewOutcome.APPROVED:
            return DecisionGovernanceDecision(
                disposition=DecisionGovernanceDisposition.ALLOW,
                decision_ref=authoritative_decision_ref(evaluation_input.decision),
                action=self.action,
                policy_context=self.policy_context,
                tenant_id=evaluation_input.decision.identity.tenant_id,
            )
        return DecisionGovernanceDecision(
            disposition=DecisionGovernanceDisposition.REQUIRE_HUMAN,
            decision_ref=authoritative_decision_ref(evaluation_input.decision),
            action=self.action,
            policy_context=self.policy_context,
            tenant_id=evaluation_input.decision.identity.tenant_id,
        )


@dataclass(frozen=True, slots=True)
class Mp4R7PostHumanDenyGovernanceEvaluator:
    """Qualification evaluator: REQUIRE_HUMAN pre-HITL, DENY after approved human review."""

    action: object
    policy_context: object

    def evaluate(self, *, evaluation_input):
        human = evaluation_input.human_review_decision
        if human is not None and human.outcome is DecisionHumanReviewOutcome.APPROVED:
            return DecisionGovernanceDecision(
                disposition=DecisionGovernanceDisposition.DENY,
                decision_ref=authoritative_decision_ref(evaluation_input.decision),
                action=self.action,
                policy_context=self.policy_context,
                tenant_id=evaluation_input.decision.identity.tenant_id,
            )
        return DecisionGovernanceDecision(
            disposition=DecisionGovernanceDisposition.REQUIRE_HUMAN,
            decision_ref=authoritative_decision_ref(evaluation_input.decision),
            action=self.action,
            policy_context=self.policy_context,
            tenant_id=evaluation_input.decision.identity.tenant_id,
        )


def mp4r7_governance_spec(
    evaluator: Mp4R7RequireHumanGovernanceEvaluator | Mp4R7PostHumanDenyGovernanceEvaluator,
) -> DecisionFlowGovernanceSpec[object]:
    return DecisionFlowGovernanceSpec(
        action=evaluator.action,
        policy_context=evaluator.policy_context,
        evaluator=evaluator,
    )


def mp4r7_verification_pipeline(stage: VerificationStage[object]) -> VerificationPipeline[object]:
    kind = validate_verification_stage_kind(stage.kind)
    return VerificationPipeline(
        registry=verification_stage_registry(
            (
                VerificationStageRegistration(
                    kind=kind,
                    stage=stage,
                    required=True,
                ),
            ),
        ),
    )


def mp4r7_governance_action():
    return decision_execution_action(
        kind=validate_decision_execution_action_kind("tool.notify"),
        subject="mp4r7-enterprise",
    )


def mp4r7_governance_policy_context():
    return decision_governance_policy_context(
        policy_provenance_digest="sha256:" + ("ab" * 32),
        matched_rule_ids=("mp4r7.require_human",),
    )


def mp4r7_stale_execution_policy_context():
    """Distinct policy context for execution-time freshness qualification (not mint-time P1)."""
    return decision_governance_policy_context(
        policy_provenance_digest="sha256:" + ("cd" * 32),
        matched_rule_ids=("mp4r7.stale_execution",),
    )


__all__ = [
    "Mp4R7PassedVerificationStage",
    "Mp4R7RecordingHumanReviewPort",
    "Mp4R7PostHumanDenyGovernanceEvaluator",
    "Mp4R7RequireHumanGovernanceEvaluator",
    "mp4r7_governance_action",
    "mp4r7_governance_policy_context",
    "mp4r7_stale_execution_policy_context",
    "mp4r7_governance_spec",
    "mp4r7_verification_pipeline",
]
