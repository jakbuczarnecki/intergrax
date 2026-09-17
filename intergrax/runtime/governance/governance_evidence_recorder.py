# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Record Governance decisions as durable evidence without altering decisions (GR-8)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceDecisionEvidenceFact,
    GovernanceEvidencePersistenceOutcome,
    GovernanceEvidencePersistencePort,
)


@dataclass(frozen=True, slots=True)
class GovernanceEvidenceRecorder:
    """Non-authoritative projection helper — failures do not change Governance outcomes."""

    persistence: GovernanceEvidencePersistencePort | None

    def record(self, fact: GovernanceDecisionEvidenceFact) -> GovernanceEvidencePersistenceOutcome | None:
        if self.persistence is None:
            return None
        return self.persistence.persist(fact)


__all__ = ["GovernanceEvidenceRecorder"]
