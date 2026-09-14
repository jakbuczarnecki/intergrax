# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""EE-B1.1 — explicit persistence failure reactions (decision-only)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.contracts.execution_evidence.persistence_failure_contract import (
    EvidencePersistenceFailureCategory,
)
from intergrax.contracts.execution_evidence.persistence_reliability_policy_contract import (
    PersistenceReliabilityDisposition,
    RuntimeEvidenceDurabilityRequirement,
)

__all__ = [
    "PersistenceFailurePolicy",
    "PersistenceFailureSurface",
    "resolve_persistence_failure_policy",
]


class PersistenceFailureSurface(StrEnum):
    """Execution-owned persistence surfaces with explicit semantics."""

    CHECKPOINT_SAVE = "checkpoint_save"
    EVENT_APPEND = "event_append"
    STATE_PERSISTENCE = "state_persistence"
    LINEAGE_PERSISTENCE = "lineage_persistence"


class PersistenceFailurePolicy(StrEnum):
    """Enterprise persistence failure reaction (maps to runtime resilience controls)."""

    FAIL_CLOSED = "fail_closed"
    RETRY = "retry"
    DEGRADE = "degrade"
    ESCALATE = "escalate"


def resolve_persistence_failure_policy(
    *,
    policy: PersistenceFailurePolicy,
    durability: RuntimeEvidenceDurabilityRequirement,
    category: EvidencePersistenceFailureCategory,
) -> PersistenceReliabilityDisposition:
    """
    Map EE-B1.1 policy to runtime disposition.

    Mandatory durability never degrades to success-without-evidence.
    Integrity failures on checkpoints never allow resume-with-corruption.
    """
    if durability is RuntimeEvidenceDurabilityRequirement.MANDATORY:
        if policy is PersistenceFailurePolicy.DEGRADE:
            return PersistenceReliabilityDisposition.FAIL_CLOSED
        if policy in {
            PersistenceFailurePolicy.FAIL_CLOSED,
            PersistenceFailurePolicy.ESCALATE,
            PersistenceFailurePolicy.RETRY,
        }:
            return PersistenceReliabilityDisposition.FAIL_CLOSED
    if category is EvidencePersistenceFailureCategory.INTEGRITY:
        return PersistenceReliabilityDisposition.FAIL_CLOSED
    if policy is PersistenceFailurePolicy.DEGRADE:
        return PersistenceReliabilityDisposition.ALLOW_CONTINUE
    if policy is PersistenceFailurePolicy.FAIL_CLOSED:
        return PersistenceReliabilityDisposition.FAIL_CLOSED
    if policy is PersistenceFailurePolicy.ESCALATE:
        return PersistenceReliabilityDisposition.FAIL_CLOSED
    if policy is PersistenceFailurePolicy.RETRY:
        return PersistenceReliabilityDisposition.FAIL_CLOSED
    return PersistenceReliabilityDisposition.FAIL_CLOSED
