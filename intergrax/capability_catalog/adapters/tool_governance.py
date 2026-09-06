# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool policy/access governance adapter for capability discovery (Stage 5)."""

from __future__ import annotations

from typing import Final

from intergrax.capability_catalog.governance import CapabilityGovernanceDecision
from intergrax.capability_catalog.ranked_candidate import RankedCapabilityCandidate
from intergrax.contracts.capability_catalog.governance import (
    CapabilityGovernanceContext,
    CapabilityGovernancePosture,
    CapabilityGovernanceReasonCode,
    GovernanceDecisionEvidence,
    GovernanceDisposition,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind

TOOL_POLICY_GOVERNANCE_EVALUATOR_ID: Final = "tool.policy.projection"


def _allowed_decision(evaluator_id: str) -> CapabilityGovernanceDecision:
    return CapabilityGovernanceDecision(
        disposition=GovernanceDisposition.ALLOWED,
        evidence=GovernanceDecisionEvidence(
            evaluator_id=evaluator_id,
            disposition=GovernanceDisposition.ALLOWED,
            reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_NOT_APPLICABLE,
        ),
    )


class ToolPolicyGovernanceEvaluator:
    """Projects caller-supplied Tool access evidence — no execution or policy engine."""

    @property
    def evaluator_id(self) -> str:
        return TOOL_POLICY_GOVERNANCE_EVALUATOR_ID

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        if candidate.identity.kind is not CapabilityKind.TOOL:
            return _allowed_decision(self.evaluator_id)

        tool_evidence = context.tool_evidence
        if tool_evidence is None:
            if context.posture is CapabilityGovernancePosture.STRICT:
                return CapabilityGovernanceDecision(
                    disposition=GovernanceDisposition.BLOCKED,
                    evidence=GovernanceDecisionEvidence(
                        evaluator_id=self.evaluator_id,
                        disposition=GovernanceDisposition.BLOCKED,
                        reason_code=CapabilityGovernanceReasonCode.MISSING_REQUIRED_EVIDENCE,
                        detail="tool governance evidence required in STRICT posture",
                    ),
                )
            return _allowed_decision(self.evaluator_id)

        identity_key = CapabilityIdentityKey.from_discovery_identity(candidate.identity)
        sort_key = identity_key.sort_key
        denied = {key.sort_key for key in tool_evidence.denied_keys}
        allowed = {key.sort_key for key in tool_evidence.allowed_keys}

        if sort_key in denied and sort_key in allowed:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=(
                        CapabilityGovernanceReasonCode.CONFLICTING_GOVERNANCE_EVIDENCE
                    ),
                    detail="tool identity appears in both allowed and denied evidence",
                ),
            )

        if sort_key in denied:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.POLICY_DENIED,
                ),
            )

        if allowed and sort_key not in allowed:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.NOT_ENTITLED,
                ),
            )

        return CapabilityGovernanceDecision(
            disposition=GovernanceDisposition.ALLOWED,
            evidence=GovernanceDecisionEvidence(
                evaluator_id=self.evaluator_id,
                disposition=GovernanceDisposition.ALLOWED,
                reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_ALLOWED,
            ),
        )
