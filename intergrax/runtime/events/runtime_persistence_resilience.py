# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution Runtime resilience controls around ``EvidencePersistencePort`` failures."""

from __future__ import annotations

from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    EvidencePersistenceBoundaryError,
    EvidencePersistenceIntegrityError,
    MandatoryEvidencePersistenceError,
)
from intergrax.contracts.execution_evidence.persistence_failure_contract import (
    ControlledEvidencePersistenceFailure,
    EvidencePersistenceFailureCategory,
)
from intergrax.contracts.execution_evidence.persistence_reliability_policy_contract import (
    KnownPersistencePortFailure,
    PersistenceReliabilityDisposition,
    PersistenceReliabilityPolicy,
    PersistenceReliabilityPolicyRequest,
    RuntimeEvidenceDurabilityRequirement,
)
from intergrax.runtime.events.default_persistence_reliability_policy import (
    default_persistence_reliability_policy,
)
from intergrax.runtime.events.evidence_durability import EvidencePersistenceRequirement
from intergrax.runtime.events.runtime_event import RuntimeEventType

__all__ = [
    "classify_controlled_persistence_failure",
    "resolve_runtime_persistence_failure",
]


def _failure_category(failure: EvidencePersistenceBoundaryError) -> EvidencePersistenceFailureCategory:
    if isinstance(failure, EvidencePersistenceIntegrityError):
        return EvidencePersistenceFailureCategory.INTEGRITY
    return EvidencePersistenceFailureCategory.INFRASTRUCTURE


def _durability_requirement(
    requirement: EvidencePersistenceRequirement,
) -> RuntimeEvidenceDurabilityRequirement:
    if requirement is EvidencePersistenceRequirement.BEST_EFFORT:
        return RuntimeEvidenceDurabilityRequirement.BEST_EFFORT
    return RuntimeEvidenceDurabilityRequirement.MANDATORY


def _policy_request(
    *,
    failure: EvidencePersistenceBoundaryError,
    requirement: EvidencePersistenceRequirement,
) -> PersistenceReliabilityPolicyRequest:
    return PersistenceReliabilityPolicyRequest(
        failure=KnownPersistencePortFailure(category=_failure_category(failure)),
        durability=_durability_requirement(requirement),
    )


def classify_controlled_persistence_failure(
    failure: EvidencePersistenceBoundaryError,
    *,
    requirement: EvidencePersistenceRequirement,
    policy: PersistenceReliabilityPolicy | None = None,
) -> ControlledEvidencePersistenceFailure:
    """Map a port-boundary error to a controlled runtime persistence failure."""
    request = _policy_request(failure=failure, requirement=requirement)
    active_policy = policy or default_persistence_reliability_policy()
    decision = active_policy.decide(request)
    return ControlledEvidencePersistenceFailure(
        category=request.failure.category,
        runtime_may_continue=(
            decision.disposition is PersistenceReliabilityDisposition.ALLOW_CONTINUE
        ),
    )


def resolve_runtime_persistence_failure(
    *,
    requirement: EvidencePersistenceRequirement,
    failure: EvidencePersistenceBoundaryError,
    event_type: RuntimeEventType,
    policy: PersistenceReliabilityPolicy | None = None,
) -> None:
    """
    Enforce runtime resilience policy for a persistence port failure.

    Raises ``MandatoryEvidencePersistenceError`` when execution must fail closed.
    Returns normally when the runtime may continue after a controlled failure.
    """
    if isinstance(failure, MandatoryEvidencePersistenceError):
        raise failure
    controlled = classify_controlled_persistence_failure(
        failure,
        requirement=requirement,
        policy=policy,
    )
    if not controlled.runtime_may_continue:
        raise MandatoryEvidencePersistenceError(
            "mandatory runtime event evidence persistence failed for "
            f"{event_type.value}",
        ) from failure
