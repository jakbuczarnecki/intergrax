# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill profile governance adapter for capability discovery (Stage 5)."""

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

SKILL_PROFILE_GOVERNANCE_EVALUATOR_ID: Final = "skill.profile.projection"


def _allowed_decision(evaluator_id: str) -> CapabilityGovernanceDecision:
    return CapabilityGovernanceDecision(
        disposition=GovernanceDisposition.ALLOWED,
        evidence=GovernanceDecisionEvidence(
            evaluator_id=evaluator_id,
            disposition=GovernanceDisposition.ALLOWED,
            reason_code=CapabilityGovernanceReasonCode.GOVERNANCE_NOT_APPLICABLE,
        ),
    )


class SkillProfileGovernanceEvaluator:
    """Projects caller-supplied Skill profile evidence — no new Skill authority."""

    @property
    def evaluator_id(self) -> str:
        return SKILL_PROFILE_GOVERNANCE_EVALUATOR_ID

    def evaluate(
        self,
        candidate: RankedCapabilityCandidate,
        context: CapabilityGovernanceContext,
    ) -> CapabilityGovernanceDecision:
        if candidate.identity.kind is not CapabilityKind.SKILL:
            return _allowed_decision(self.evaluator_id)

        skill_evidence = context.skill_evidence
        if skill_evidence is None:
            if context.posture is CapabilityGovernancePosture.STRICT:
                return CapabilityGovernanceDecision(
                    disposition=GovernanceDisposition.BLOCKED,
                    evidence=GovernanceDecisionEvidence(
                        evaluator_id=self.evaluator_id,
                        disposition=GovernanceDisposition.BLOCKED,
                        reason_code=CapabilityGovernanceReasonCode.MISSING_REQUIRED_EVIDENCE,
                        detail="skill governance evidence required in STRICT posture",
                    ),
                )
            return _allowed_decision(self.evaluator_id)

        identity_key = CapabilityIdentityKey.from_discovery_identity(candidate.identity)
        sort_key = identity_key.sort_key
        enabled = {key.sort_key for key in skill_evidence.enabled_keys}
        blocked = {key.sort_key for key in skill_evidence.blocked_keys}

        if sort_key in blocked and sort_key in enabled:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=(
                        CapabilityGovernanceReasonCode.CONFLICTING_GOVERNANCE_EVIDENCE
                    ),
                    detail="skill identity appears in both enabled and blocked evidence",
                ),
            )

        if sort_key in blocked:
            return CapabilityGovernanceDecision(
                disposition=GovernanceDisposition.BLOCKED,
                evidence=GovernanceDecisionEvidence(
                    evaluator_id=self.evaluator_id,
                    disposition=GovernanceDisposition.BLOCKED,
                    reason_code=CapabilityGovernanceReasonCode.POLICY_DENIED,
                ),
            )

        if enabled and sort_key not in enabled:
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
