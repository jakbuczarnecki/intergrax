# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent trust/admission governance adapter for capability discovery (Stage 5)."""

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

AGENT_TRUST_GOVERNANCE_EVALUATOR_ID: Final = "agent.trust.projection"


def _allowed_decision(evaluator_id: str) -> CapabilityGovernanceDecision:
    return CapabilityGovernanceDecision(
        disposition=GovernanceDisposition.ALLOWED,
        evidence=GovernanceDecisionEvidence(
            evaluator_id=evaluator_id,
            disposition=GovernanceDisposition.ALLOWED,
            reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_NOT_APPLICABLE,
        ),
    )


class AgentTrustGovernanceEvaluator:
    """Projects caller-supplied trust/admission evidence — no trust verification."""

    @property
    def evaluator_id(self) -> str:
        return AGENT_TRUST_GOVERNANCE_EVALUATOR_ID

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        if candidate.identity.kind is not CapabilityKind.AGENT:
            return _allowed_decision(self.evaluator_id)

        agent_evidence = context.agent_evidence
        if agent_evidence is None:
            if context.posture is CapabilityGovernancePosture.STRICT:
                return CapabilityGovernanceDecision(
                    disposition=GovernanceDisposition.BLOCKED,
                    evidence=GovernanceDecisionEvidence(
                        evaluator_id=self.evaluator_id,
                        disposition=GovernanceDisposition.BLOCKED,
                        reason_code=CapabilityGovernanceReasonCode.MISSING_REQUIRED_EVIDENCE,
                        detail="agent governance evidence required in STRICT posture",
                    ),
                )
            return _allowed_decision(self.evaluator_id)

        identity_key = CapabilityIdentityKey.from_discovery_identity(candidate.identity)
        sort_key = identity_key.sort_key
        trusted = {key.sort_key for key in agent_evidence.trusted_keys}
        blocked = {key.sort_key for key in agent_evidence.blocked_keys}
        revoked = {key.sort_key for key in agent_evidence.revoked_keys}

        if sort_key in blocked and sort_key in trusted:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=(
                        CapabilityGovernanceReasonCode.CONFLICTING_GOVERNANCE_EVIDENCE
                    ),
                    detail="agent identity appears in both trusted and blocked evidence",
                ),
            )

        if sort_key in revoked or sort_key in blocked:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.TRUST_NOT_SATISFIED,
                ),
            )

        if trusted and sort_key not in trusted:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.TRUST_NOT_SATISFIED,
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
