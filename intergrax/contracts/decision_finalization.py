# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision System finalization guard contracts (DS-CORE-05).

Pure domain guard enforcing DS-INV-003: at most one terminal authoritative
outcome per decision scope. Durable atomic compare-and-set / uniqueness
belongs to the canonical hosting Execution persistence/checkpoint boundary
(DS-CORE-06+); Nexus participates only when the hosting Execution uses
orchestration. This module does not provide storage, locks, or cross-process
concurrency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from intergrax.contracts.decision_identity import (
    DecisionId,
    DecisionIdentity,
    DecisionScope,
    validate_decision_id,
    validate_decision_tenant_id,
)
from intergrax.contracts.decision_record import AuthoritativeAcceptedDecision
from intergrax.contracts.decision_resolution import AuthoritativeResolutionRecord

T = TypeVar("T")


class DecisionFinalizeDisposition(str, Enum):
    """Outcome of one guard evaluation — not a lifecycle stage."""

    FIRST_FINALIZATION = "first_finalization"
    IDEMPOTENT_REPLAY = "idempotent_replay"


class DecisionFinalizationConflictError(ValueError):
    """Raised when a different terminal authoritative outcome already exists."""


@dataclass(frozen=True, slots=True)
class DecisionFinalizationKey:
    """Stable authority scope for finalize uniqueness (DS-INV-003).

    Intentionally excludes ``DecisionVersion`` and execution retry lineage
    (``AttemptId``, ``RunId``, ``ExecutionId``). Version and execution
  remain on the authoritative record for audit; retries must target this key.
    """

    decision_id: DecisionId
    scope: DecisionScope
    tenant_id: str

    def __post_init__(self) -> None:
        validate_decision_id(self.decision_id)
        if type(self.scope) is not DecisionScope:
            raise TypeError("DecisionFinalizationKey.scope must be DecisionScope")
        validate_decision_tenant_id(self.tenant_id)


def decision_finalization_key(identity: DecisionIdentity) -> DecisionFinalizationKey:
    """Derive finalize key from identity, ignoring version and execution lineage."""
    if type(identity) is not DecisionIdentity:
        raise TypeError("identity must be DecisionIdentity")
    return DecisionFinalizationKey(
        decision_id=identity.decision_id,
        scope=identity.scope,
        tenant_id=identity.tenant_id,
    )


@dataclass(frozen=True, slots=True)
class DecisionFinalizeGuardState(Generic[T]):
    """Immutable in-memory guard position for one finalization scope."""

    key: DecisionFinalizationKey
    authoritative_outcome: (
        AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord | None
    )


def initial_decision_finalize_guard[T](
    key: DecisionFinalizationKey,
) -> DecisionFinalizeGuardState[T]:
    """Return an unfinalized guard for one decision finalization scope."""
    if type(key) is not DecisionFinalizationKey:
        raise TypeError("key must be DecisionFinalizationKey")
    return DecisionFinalizeGuardState(key=key, authoritative_outcome=None)


@dataclass(frozen=True, slots=True)
class DecisionFinalizeGuardResult(Generic[T]):
    """Typed result of one pure guard evaluation."""

    state: DecisionFinalizeGuardState[T]
    disposition: DecisionFinalizeDisposition


def guard_decision_finalization(
    state: DecisionFinalizeGuardState[T],
    requested_outcome: AuthoritativeAcceptedDecision[T] | AuthoritativeResolutionRecord,
) -> DecisionFinalizeGuardResult[T]:
    """Evaluate whether a terminal authoritative outcome may be committed.

    First finalization stores the requested outcome. Exact replay of the same
    immutable authoritative record is idempotent. A different terminal outcome
    for the same key fails closed.

    Idempotent replay requires the **exact** persisted authoritative record.
    Reconstructing a semantically similar record with new execution lineage
    (e.g. a new ``AttemptId`` after crash) is **not** replay — persistence
    integration must resume the stored outcome (DS-CORE-06).
    """
    if type(state) is not DecisionFinalizeGuardState:
        raise TypeError("state must be DecisionFinalizeGuardState")
    if type(requested_outcome) not in (
        AuthoritativeAcceptedDecision,
        AuthoritativeResolutionRecord,
    ):
        raise TypeError(
            "requested_outcome must be AuthoritativeAcceptedDecision "
            "or AuthoritativeResolutionRecord",
        )

    requested_key = decision_finalization_key(requested_outcome.identity)
    if requested_key != state.key:
        raise ValueError(
            "requested outcome finalization key does not match guard state key",
        )

    existing = state.authoritative_outcome
    if existing is None:
        new_state = DecisionFinalizeGuardState(
            key=state.key,
            authoritative_outcome=requested_outcome,
        )
        return DecisionFinalizeGuardResult(
            state=new_state,
            disposition=DecisionFinalizeDisposition.FIRST_FINALIZATION,
        )

    if existing == requested_outcome:
        return DecisionFinalizeGuardResult(
            state=state,
            disposition=DecisionFinalizeDisposition.IDEMPOTENT_REPLAY,
        )

    key = state.key
    raise DecisionFinalizationConflictError(
        "Conflicting terminal authoritative outcome for "
        f"decision_id={key.decision_id!r}, "
        f"tenant_id={key.tenant_id!r}, "
        f"scope={key.scope.namespace!r}/{key.scope.subject!r}",
    )
