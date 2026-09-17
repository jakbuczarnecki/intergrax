# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition helpers for Governance evidence persistence (GR-8)."""

from __future__ import annotations

from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceEvidencePersistencePort,
)
from intergrax.runtime.governance.governance_evidence_persistence import (
    InMemoryGovernanceEvidencePersistence,
    RuntimeEventGovernanceEvidencePersistence,
)
from intergrax.runtime.governance.governance_evidence_recorder import GovernanceEvidenceRecorder


def build_governance_evidence_recorder(
    *,
    persistence: GovernanceEvidencePersistencePort | None,
) -> GovernanceEvidenceRecorder:
    return GovernanceEvidenceRecorder(persistence=persistence)


def build_runtime_event_governance_evidence_persistence(
    *,
    evidence_persistence: EvidencePersistencePort,
) -> RuntimeEventGovernanceEvidencePersistence:
    return RuntimeEventGovernanceEvidencePersistence(
        evidence_persistence=evidence_persistence,
    )


def build_in_memory_governance_evidence_persistence() -> InMemoryGovernanceEvidencePersistence:
    return InMemoryGovernanceEvidencePersistence()


__all__ = [
    "build_governance_evidence_recorder",
    "build_in_memory_governance_evidence_persistence",
    "build_runtime_event_governance_evidence_persistence",
]
