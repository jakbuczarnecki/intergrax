# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Diagnostics contract for persistence reliability decisions (not execution evidence)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_evidence.persistence_failure_contract import (
    EvidencePersistenceFailureCategory,
)
from intergrax.contracts.execution_evidence.persistence_reliability_policy_contract import (
    PersistenceReliabilityDisposition,
    PersistenceReliabilityPolicy,
    PersistenceReliabilityPolicyDecision,
    PersistenceReliabilityPolicyRequest,
    RuntimeEvidenceDurabilityRequirement,
)

__all__ = [
    "ANONYMOUS_PERSISTENCE_RELIABILITY_POLICY_ID",
    "NullPersistenceReliabilityDecisionObserver",
    "PersistenceReliabilityDecisionObserver",
    "PersistenceReliabilityDiagnostic",
    "PersistenceReliabilityPolicyIdentifiable",
    "build_persistence_reliability_diagnostic",
    "resolve_persistence_reliability_policy_id",
]

ANONYMOUS_PERSISTENCE_RELIABILITY_POLICY_ID = "anonymous_persistence_reliability_policy"


@dataclass(frozen=True, slots=True)
class PersistenceReliabilityDiagnostic:
    """
    Typed view of a persistence reliability decision for operators and adapters.

    Describes what problem was reported, which policy reacted, and the chosen
    disposition. This is not execution evidence and must not be persisted as such.
    """

    failure_category: EvidencePersistenceFailureCategory
    durability: RuntimeEvidenceDurabilityRequirement
    disposition: PersistenceReliabilityDisposition
    policy_id: str
    runtime_event_type: str | None


@runtime_checkable
class PersistenceReliabilityPolicyIdentifiable(Protocol):
    """Optional policy identity for diagnostics and audit correlation."""

    @property
    def policy_id(self) -> str:
        """Stable, provider-agnostic policy identifier."""


class PersistenceReliabilityDecisionObserver(Protocol):
    """Pluggable sink for reliability decision diagnostics (logger, monitoring, etc.)."""

    def observe_persistence_reliability_decision(
        self,
        diagnostic: PersistenceReliabilityDiagnostic,
    ) -> None:
        """Receive a reliability decision diagnostic; must not mutate execution state."""


class NullPersistenceReliabilityDecisionObserver:
    """Default no-op observer preserving existing runtime behavior."""

    def observe_persistence_reliability_decision(
        self,
        diagnostic: PersistenceReliabilityDiagnostic,
    ) -> None:
        return None


def resolve_persistence_reliability_policy_id(
    policy: PersistenceReliabilityPolicy,
) -> str:
    if isinstance(policy, PersistenceReliabilityPolicyIdentifiable):
        return policy.policy_id
    return ANONYMOUS_PERSISTENCE_RELIABILITY_POLICY_ID


def build_persistence_reliability_diagnostic(
    *,
    request: PersistenceReliabilityPolicyRequest,
    decision: PersistenceReliabilityPolicyDecision,
    policy: PersistenceReliabilityPolicy,
    runtime_event_type: str | None = None,
) -> PersistenceReliabilityDiagnostic:
    return PersistenceReliabilityDiagnostic(
        failure_category=request.failure.category,
        durability=request.durability,
        disposition=decision.disposition,
        policy_id=resolve_persistence_reliability_policy_id(policy),
        runtime_event_type=runtime_event_type,
    )
