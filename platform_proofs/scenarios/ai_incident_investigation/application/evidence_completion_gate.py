# © Artur Czarnecki. All rights reserved.

"""Production completion eligibility gate for AI Incident (DS-E2E-15B.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from intergrax.decision_system.completion_eligibility import (
    CompletionEligibilityDecision,
    CompletionEligibilityStatus,
    evaluate_completion_eligibility_with_provider,
)
from intergrax.decision_system.evidence_requirements import EvidenceRequirementProvider
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_requirement_semantics import (
    RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    project_resolved_incident_evidence_assessment,
)


class CompletionEligibilityBlockedError(Exception):
    """Raised when mandatory evidence blocks terminal completion acceptance."""

    def __init__(self, decision: CompletionEligibilityDecision) -> None:
        self.decision = decision
        unresolved = ", ".join(
            str(requirement_id)
            for requirement_id in decision.unresolved_mandatory_requirement_ids
        )
        super().__init__(
            "completion_ineligible_unresolved_mandatory_evidence:"
            f"{unresolved or 'none'}"
        )


@dataclass(frozen=True, slots=True)
class CompletionEligibilityGateConfig:
    enabled: bool = True
    provider: EvidenceRequirementProvider | None = RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER


def evaluate_ai_incident_completion_eligibility(
    *,
    evidence_nodes: Sequence[Mapping[str, object]],
    provider: EvidenceRequirementProvider | None,
) -> CompletionEligibilityDecision:
    assessment = project_resolved_incident_evidence_assessment(evidence_nodes)
    return evaluate_completion_eligibility_with_provider(
        provider=provider,
        assessment=assessment,
    )


def assert_ai_incident_completion_eligible(
    *,
    evidence_nodes: Sequence[Mapping[str, object]],
    gate: CompletionEligibilityGateConfig,
) -> CompletionEligibilityDecision | None:
    """Enforce completion eligibility before terminal acceptance when gate is enabled."""
    if not gate.enabled:
        return None
    decision = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=evidence_nodes,
        provider=gate.provider,
    )
    if decision.status is CompletionEligibilityStatus.INELIGIBLE:
        raise CompletionEligibilityBlockedError(decision)
    return decision
