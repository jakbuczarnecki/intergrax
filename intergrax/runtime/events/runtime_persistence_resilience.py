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
from intergrax.runtime.events.evidence_durability import EvidencePersistenceRequirement
from intergrax.runtime.events.runtime_event import RuntimeEventType

__all__ = [
    "classify_controlled_persistence_failure",
    "resolve_runtime_persistence_failure",
]


def classify_controlled_persistence_failure(
    failure: EvidencePersistenceBoundaryError,
    *,
    requirement: EvidencePersistenceRequirement,
) -> ControlledEvidencePersistenceFailure:
    """Map a port-boundary error to a controlled runtime persistence failure."""
    if isinstance(failure, EvidencePersistenceIntegrityError):
        category = EvidencePersistenceFailureCategory.INTEGRITY
    else:
        category = EvidencePersistenceFailureCategory.INFRASTRUCTURE
    runtime_may_continue = requirement is EvidencePersistenceRequirement.BEST_EFFORT
    return ControlledEvidencePersistenceFailure(
        category=category,
        runtime_may_continue=runtime_may_continue,
    )


def resolve_runtime_persistence_failure(
    *,
    requirement: EvidencePersistenceRequirement,
    failure: EvidencePersistenceBoundaryError,
    event_type: RuntimeEventType,
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
    )
    if not controlled.runtime_may_continue:
        raise MandatoryEvidencePersistenceError(
            "mandatory runtime event evidence persistence failed for "
            f"{event_type.value}",
        ) from failure
