# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Run-scoped Decision exposure candidate collector (execution host infrastructure)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidate,
    DecisionExposureCandidateAppend,
)
from intergrax.contracts.execution_identity import AttemptId, validate_attempt_id

T = TypeVar("T")


class DecisionExposureCollectorError(ValueError):
    """Fail-closed collector invariant violation."""


@dataclass(frozen=True, slots=True)
class _CandidatePartitionKey:
    evaluation_scope: object
    namespace: str
    subject: str
    host_publication_class: object


class DecisionExposureCandidateCollector(Generic[T]):
    """In-memory collector partitioned by AttemptId (instance-scoped only)."""

    def __init__(self) -> None:
        self._partitions: dict[AttemptId, list[DecisionExposureCandidate[T]]] = {}
        self._next_ordinal: dict[AttemptId, int] = {}
        self._keys: dict[AttemptId, set[_CandidatePartitionKey]] = {}

    def append(self, fragment: DecisionExposureCandidateAppend[T]) -> None:
        """Append one candidate; assigns monotonic evaluation_ordinal per attempt."""
        if type(fragment) is not DecisionExposureCandidateAppend:
            raise TypeError("fragment must be DecisionExposureCandidateAppend")
        attempt_id = validate_attempt_id(fragment.execution_lineage.attempt_id)
        key = _CandidatePartitionKey(
            evaluation_scope=fragment.evaluation_scope,
            namespace=fragment.decision_scope.namespace,
            subject=fragment.decision_scope.subject,
            host_publication_class=fragment.host_publication_class,
        )
        keys = self._keys.setdefault(attempt_id, set())
        if key in keys:
            raise DecisionExposureCollectorError(
                "duplicate decision exposure candidate for attempt partition",
            )
        ordinal = self._next_ordinal.get(attempt_id, 0)
        self._next_ordinal[attempt_id] = ordinal + 1
        candidate = DecisionExposureCandidate(
            evaluation_scope=fragment.evaluation_scope,
            decision_scope=fragment.decision_scope,
            execution_lineage=fragment.execution_lineage,
            host_publication_class=fragment.host_publication_class,
            exposure=fragment.exposure,
            evaluation_ordinal=ordinal,
        )
        keys.add(key)
        self._partitions.setdefault(attempt_id, []).append(candidate)

    def candidates_for_attempt(self, attempt_id: AttemptId) -> tuple[DecisionExposureCandidate[T], ...]:
        """Return an immutable snapshot for one attempt partition."""
        resolved = validate_attempt_id(attempt_id)
        stored = self._partitions.get(resolved, ())
        return tuple(stored)

    def count(self, attempt_id: AttemptId) -> int:
        """Return candidate count for one attempt partition."""
        resolved = validate_attempt_id(attempt_id)
        return len(self._partitions.get(resolved, ()))
