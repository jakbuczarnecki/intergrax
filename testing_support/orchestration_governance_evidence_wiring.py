# © Artur Czarnecki. All rights reserved.

"""Test composition helpers for orchestration Governance Evidence (GR-10-R14)."""

from __future__ import annotations

from intergrax.contracts.governed_execution_governance_evidence import (
    GovernanceEvidencePersistencePort,
)
from intergrax.runtime.governance.governance_evidence_composition import (
    build_in_memory_governance_evidence_persistence,
)


def default_test_orchestration_evidence_persistence() -> GovernanceEvidencePersistencePort:
    """Explicit test composition — in-memory GR-8 port (not production default)."""
    return build_in_memory_governance_evidence_persistence()


__all__ = ["default_test_orchestration_evidence_persistence"]
