# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default persistence reliability policy (current production semantics)."""

from __future__ import annotations

from intergrax.contracts.execution_evidence.persistence_reliability_policy_contract import (
    PersistenceReliabilityDisposition,
    PersistenceReliabilityPolicy,
    PersistenceReliabilityPolicyDecision,
    PersistenceReliabilityPolicyRequest,
    RuntimeEvidenceDurabilityRequirement,
)

__all__ = [
    "DEFAULT_PERSISTENCE_RELIABILITY_POLICY_ID",
    "DefaultPersistenceReliabilityPolicy",
    "default_persistence_reliability_policy",
]

DEFAULT_PERSISTENCE_RELIABILITY_POLICY_ID = "default_persistence_reliability"


class DefaultPersistenceReliabilityPolicy:
    """Durability-based fail-closed vs best-effort continue (legacy runtime behavior)."""

    @property
    def policy_id(self) -> str:
        return DEFAULT_PERSISTENCE_RELIABILITY_POLICY_ID

    def decide(
        self,
        request: PersistenceReliabilityPolicyRequest,
    ) -> PersistenceReliabilityPolicyDecision:
        if request.durability is RuntimeEvidenceDurabilityRequirement.BEST_EFFORT:
            return PersistenceReliabilityPolicyDecision(
                disposition=PersistenceReliabilityDisposition.ALLOW_CONTINUE,
            )
        return PersistenceReliabilityPolicyDecision(
            disposition=PersistenceReliabilityDisposition.FAIL_CLOSED,
        )


_DEFAULT_POLICY = DefaultPersistenceReliabilityPolicy()


def default_persistence_reliability_policy() -> PersistenceReliabilityPolicy:
    """Shared default policy instance for runtime wiring."""
    return _DEFAULT_POLICY
