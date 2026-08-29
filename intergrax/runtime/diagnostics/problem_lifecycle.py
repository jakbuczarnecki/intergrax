# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""
Stable Problem identity and lifecycle over validated grouping candidates (DIAG-5D).

Three distinct concepts (never collapse):

A. candidate membership — "these executions are grouped now" (ephemeral hypothesis)
B. stable Problem identity — "this recurring pattern is the same tracked problem"
C. root cause — "these incidents are caused by X" (evidence-based, future work)

``ProblemId`` denotes B only. Grouping remains hypothesis-producing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NewType, Protocol, runtime_checkable
from uuid import uuid4

from intergrax.runtime.diagnostics.deterministic_problem_reconciliation import (
    DeterministicProblemReconciliationKey,
    ProblemReconciliationKeyKind,
    extract_deterministic_reconciliation_key,
)
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingBasisKind,
    ProblemGroupingCandidate,
    ProblemGroupingIntegrityError,
    ProblemGroupingMethod,
    ProblemGroupingProvenance,
    ProblemGroupingResult,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyVersion,
    ProblemGroupingSubjectRef,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    RECONCILIATION_WINNER_CANONICAL_PENDING,
    ProblemPersistence,
    ProblemPersistenceConflictError,
    ProblemPersistenceIntegrityError,
)

ProblemId = NewType("ProblemId", str)

_CANONICAL_SUFFIX = re.compile(r"^[0-9a-f]{32}$")
_MAX_PERSISTENCE_CONFLICT_RETRIES = 3


class ProblemLifecycleIntegrityError(Exception):
    """Raised when reconciliation input or attachment rules are violated."""


class ProblemStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


@runtime_checkable
class ProblemReconciliationKey(Protocol):
    """Strategy-specific recurrence evidence — not opaque Problem identity."""

    @property
    def kind(self) -> ProblemReconciliationKeyKind: ...

    def index_token(self) -> str: ...


@dataclass(frozen=True, slots=True)
class ProblemOccurrence:
    """Immutable accepted observation tying one execution to a stable Problem."""

    subject_ref: ProblemGroupingSubjectRef
    observed_at: datetime
    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    method: ProblemGroupingMethod


@dataclass(frozen=True, slots=True)
class ProblemLifecycleProvenance:
    """Audit trail for which strategy established or last updated a Problem."""

    strategy_id: ProblemGroupingStrategyId
    strategy_version: ProblemGroupingStrategyVersion
    method: ProblemGroupingMethod
    reconciliation_key: ProblemReconciliationKey


@dataclass(frozen=True, slots=True)
class Problem:
    """
    Persisted derived operational diagnostic state — NOT canonical execution truth.

    Rebuildable in principle from canonical evidence and validated grouping output.

    ``first_seen_at`` / ``last_seen_at`` are min/max of accepted
    ``ProblemOccurrence.observed_at`` — not lifecycle mutation times.
    Processing order and explicit resolution do not change them.
    """

    problem_id: ProblemId
    tenant_id: str
    status: ProblemStatus
    first_seen_at: datetime
    last_seen_at: datetime
    occurrence_count: int
    current_subject_refs: tuple[ProblemGroupingSubjectRef, ...]
    occurrences: tuple[ProblemOccurrence, ...]
    provenance: ProblemLifecycleProvenance
    record_version: int = 1


@dataclass(frozen=True, slots=True)
class ProblemLifecycleResult:
    """Typed reconciliation outcome for one validated grouping invocation."""

    created: tuple[Problem, ...]
    updated: tuple[Problem, ...]
    unchanged: tuple[Problem, ...]


@runtime_checkable
class ProblemReconciliationPolicy(Protocol):
    """Strategy-specific reconciliation key extraction without lifecycle coupling."""

    @property
    def supported_basis_kind(self) -> ProblemGroupingBasisKind: ...

    def extract_reconciliation_key(
        self,
        candidate: ProblemGroupingCandidate,
        *,
        tenant_id: str,
    ) -> ProblemReconciliationKey: ...


class DeterministicProblemReconciliationPolicy:
    """Production reconciliation policy for ``DeterministicProblemGroupingStrategy``."""

    @property
    def supported_basis_kind(self) -> ProblemGroupingBasisKind:
        return ProblemGroupingBasisKind.DETERMINISTIC

    def extract_reconciliation_key(
        self,
        candidate: ProblemGroupingCandidate,
        *,
        tenant_id: str,
    ) -> ProblemReconciliationKey:
        return extract_deterministic_reconciliation_key(candidate, tenant_id=tenant_id)


def validate_problem_id(value: object) -> ProblemId:
    if type(value) is not str:
        raise TypeError(f"ProblemId must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError("ProblemId must be non-empty and not whitespace-only")
    if value != value.strip():
        raise ValueError("ProblemId must not contain leading or trailing whitespace")
    if not value.startswith("problem_"):
        raise ValueError("ProblemId must start with 'problem_'")
    suffix = value[len("problem_") :]
    if not _CANONICAL_SUFFIX.fullmatch(suffix):
        raise ValueError("ProblemId suffix must match [0-9a-f]{32}")
    return ProblemId(value)


def mint_problem_id() -> ProblemId:
    return ProblemId(f"problem_{uuid4().hex}")


def reconciliation_keys_equal(
    left: ProblemReconciliationKey,
    right: ProblemReconciliationKey,
) -> bool:
    if left.kind != right.kind:
        return False
    return left.index_token() == right.index_token()


class ProblemLifecycleEngine:
    """
    Reconcile validated grouping hypotheses into stable tenant-scoped Problems.

    Owns lookup, matching, create/update, conflict detection, and persistence.
    Does not re-run grouping logic or mutate ``ProblemGroupingEngine``.
    """

    def __init__(
        self,
        persistence: ProblemPersistence,
        *,
        reconciliation_policies: tuple[ProblemReconciliationPolicy, ...] | None = None,
    ) -> None:
        policies = reconciliation_policies or (DeterministicProblemReconciliationPolicy(),)
        self._persistence = persistence
        self._policies_by_kind = {
            policy.supported_basis_kind: policy for policy in policies
        }

    def reconcile(
        self,
        grouping_result: ProblemGroupingResult,
        *,
        observed_at: datetime,
    ) -> ProblemLifecycleResult:
        tenant_id = grouping_result.tenant_id
        _validate_observed_at(observed_at)
        _validate_grouping_result_tenant(grouping_result)

        created: list[Problem] = []
        updated: list[Problem] = []
        unchanged: list[Problem] = []

        batch_subject_owner: dict[ProblemGroupingSubjectRef, ProblemId] = {}

        for candidate in grouping_result.candidates:
            reconciliation_key = self._extract_reconciliation_key(
                candidate,
                tenant_id=tenant_id,
            )
            target_problem_id = self._resolve_target_problem_id(
                candidate=candidate,
                tenant_id=tenant_id,
                reconciliation_key=reconciliation_key,
                batch_subject_owner=batch_subject_owner,
            )

            for member in candidate.members:
                prior_owner = batch_subject_owner.get(member)
                if prior_owner is not None and prior_owner != target_problem_id:
                    raise ProblemLifecycleIntegrityError(
                        "one occurrence cannot attach to two stable Problems "
                        "in the same reconciliation operation",
                    )
                batch_subject_owner[member] = target_problem_id

            existing = self._persistence.get(
                tenant_id=tenant_id,
                problem_id=target_problem_id,
            )
            if existing is None:
                record = _create_problem(
                    problem_id=target_problem_id,
                    tenant_id=tenant_id,
                    candidate=candidate,
                    reconciliation_key=reconciliation_key,
                    observed_at=observed_at,
                )
                try:
                    persisted = self._persistence.create(record)
                except ProblemPersistenceConflictError as exc:
                    persisted, changed = self._converge_after_create_conflict(
                        tenant_id=tenant_id,
                        candidate=candidate,
                        reconciliation_key=reconciliation_key,
                        observed_at=observed_at,
                        original_exc=exc,
                    )
                    if changed:
                        updated.append(persisted)
                    else:
                        unchanged.append(persisted)
                else:
                    created.append(persisted)
                continue

            persisted, changed = self._persist_candidate_with_occ_retry(
                tenant_id=tenant_id,
                problem_id=target_problem_id,
                candidate=candidate,
                reconciliation_key=reconciliation_key,
                observed_at=observed_at,
            )
            if changed:
                updated.append(persisted)
            else:
                unchanged.append(persisted)

        return ProblemLifecycleResult(
            created=tuple(created),
            updated=tuple(updated),
            unchanged=tuple(unchanged),
        )

    def resolve(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
        resolved_at: datetime,
    ) -> Problem:
        _validate_observed_at(resolved_at)
        validated_problem_id = validate_problem_id(problem_id)
        existing = self._persistence.get(
            tenant_id=tenant_id,
            problem_id=validated_problem_id,
        )
        if existing is None:
            raise ProblemLifecycleIntegrityError("Problem does not exist for tenant scope")

        reconciliation_key = existing.provenance.reconciliation_key

        for attempt in range(_MAX_PERSISTENCE_CONFLICT_RETRIES + 1):
            _validate_persisted_problem_identity(
                existing,
                tenant_id=tenant_id,
                problem_id=validated_problem_id,
                reconciliation_key=reconciliation_key,
            )

            if existing.status is ProblemStatus.RESOLVED:
                return existing

            resolved = _build_resolved_problem(existing)
            try:
                return self._persistence.update(
                    resolved,
                    expected_version=existing.record_version,
                )
            except ProblemPersistenceConflictError as exc:
                if attempt >= _MAX_PERSISTENCE_CONFLICT_RETRIES:
                    raise ProblemLifecycleIntegrityError(
                        "failed to resolve Problem: persistence conflict "
                        "persisted after bounded optimistic concurrency retries",
                    ) from exc

                latest = self._persistence.get(
                    tenant_id=tenant_id,
                    problem_id=validated_problem_id,
                )
                if latest is None:
                    raise ProblemLifecycleIntegrityError(
                        "Problem disappeared after resolve persistence conflict",
                    ) from exc
                existing = latest

        raise AssertionError("unreachable resolve persistence conflict retry loop")

    def _converge_after_create_conflict(
        self,
        *,
        tenant_id: str,
        candidate: ProblemGroupingCandidate,
        reconciliation_key: ProblemReconciliationKey,
        observed_at: datetime,
        original_exc: ProblemPersistenceConflictError,
    ) -> tuple[Problem, bool]:
        lookup_exc: BaseException = original_exc
        attempt = 0
        pending_spins = 0
        while attempt <= _MAX_PERSISTENCE_CONFLICT_RETRIES:
            try:
                winner = self._persistence.find_by_reconciliation_key(
                    tenant_id=tenant_id,
                    reconciliation_key=reconciliation_key,
                )
            except ProblemPersistenceIntegrityError as exc:
                if str(exc) != RECONCILIATION_WINNER_CANONICAL_PENDING:
                    raise ProblemLifecycleIntegrityError(
                        "failed to create stable Problem due to persistence lookup failure",
                    ) from exc
                pending_spins += 1
                if pending_spins > _MAX_PERSISTENCE_CONFLICT_RETRIES:
                    raise ProblemLifecycleIntegrityError(
                        "failed to create stable Problem: winner not yet durable "
                        "after persistence conflict",
                    ) from exc
                lookup_exc = exc
                continue

            pending_spins = 0
            if winner is None:
                if attempt >= _MAX_PERSISTENCE_CONFLICT_RETRIES:
                    raise ProblemLifecycleIntegrityError(
                        "failed to create stable Problem due to persistence conflict",
                    ) from lookup_exc
                attempt += 1
                continue

            return self._persist_candidate_with_occ_retry(
                tenant_id=tenant_id,
                problem_id=winner.problem_id,
                candidate=candidate,
                reconciliation_key=reconciliation_key,
                observed_at=observed_at,
            )

        raise AssertionError("unreachable create conflict convergence loop")

    def _persist_candidate_with_occ_retry(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
        candidate: ProblemGroupingCandidate,
        reconciliation_key: ProblemReconciliationKey,
        observed_at: datetime,
    ) -> tuple[Problem, bool]:
        existing = self._persistence.get(
            tenant_id=tenant_id,
            problem_id=problem_id,
        )
        if existing is None:
            raise ProblemLifecycleIntegrityError(
                "stable Problem is missing during persistence reconciliation",
            )

        for attempt in range(_MAX_PERSISTENCE_CONFLICT_RETRIES + 1):
            _validate_persisted_problem_identity(
                existing,
                tenant_id=tenant_id,
                problem_id=problem_id,
                reconciliation_key=reconciliation_key,
            )

            next_record, changed = _apply_candidate_to_problem(
                existing,
                candidate=candidate,
                reconciliation_key=reconciliation_key,
                observed_at=observed_at,
            )
            if not changed:
                return existing, False

            try:
                return (
                    self._persistence.update(
                        next_record,
                        expected_version=existing.record_version,
                    ),
                    True,
                )
            except ProblemPersistenceConflictError as exc:
                if attempt >= _MAX_PERSISTENCE_CONFLICT_RETRIES:
                    raise ProblemLifecycleIntegrityError(
                        "failed to update stable Problem: persistence conflict "
                        "persisted after bounded optimistic concurrency retries",
                    ) from exc

                latest = self._persistence.get(
                    tenant_id=tenant_id,
                    problem_id=problem_id,
                )
                if latest is None:
                    raise ProblemLifecycleIntegrityError(
                        "stable Problem disappeared after persistence conflict",
                    ) from exc
                existing = latest

        raise AssertionError("unreachable persistence conflict retry loop")

    def _extract_reconciliation_key(
        self,
        candidate: ProblemGroupingCandidate,
        *,
        tenant_id: str,
    ) -> ProblemReconciliationKey:
        basis = candidate.provenance.basis
        if basis is None:
            raise ProblemLifecycleIntegrityError(
                "lifecycle reconciliation requires typed grouping basis",
            )
        policy = self._policies_by_kind.get(basis.kind)
        if policy is None:
            raise ProblemLifecycleIntegrityError(
                f"no reconciliation policy registered for basis kind {basis.kind!r}",
            )
        return policy.extract_reconciliation_key(candidate, tenant_id=tenant_id)

    def _resolve_target_problem_id(
        self,
        *,
        candidate: ProblemGroupingCandidate,
        tenant_id: str,
        reconciliation_key: ProblemReconciliationKey,
        batch_subject_owner: dict[ProblemGroupingSubjectRef, ProblemId],
    ) -> ProblemId:
        matched_ids: set[ProblemId] = set()

        by_key = self._persistence.find_by_reconciliation_key(
            tenant_id=tenant_id,
            reconciliation_key=reconciliation_key,
        )
        if by_key is not None:
            matched_ids.add(by_key.problem_id)

        for member in candidate.members:
            by_subject = self._persistence.find_by_subject_ref(
                tenant_id=tenant_id,
                subject_ref=member,
            )
            if by_subject is not None:
                matched_ids.add(by_subject.problem_id)

            batch_owner = batch_subject_owner.get(member)
            if batch_owner is not None:
                matched_ids.add(batch_owner)

        if len(matched_ids) > 1:
            raise ProblemLifecycleIntegrityError(
                "candidate maps to multiple incompatible stable Problems",
            )

        if len(matched_ids) == 1:
            (problem_id,) = tuple(matched_ids)
            existing = self._persistence.get(tenant_id=tenant_id, problem_id=problem_id)
            if existing is None:
                raise ProblemLifecycleIntegrityError(
                    "resolved Problem id is missing from persistence",
                )
            if not reconciliation_keys_equal(
                existing.provenance.reconciliation_key,
                reconciliation_key,
            ):
                raise ProblemLifecycleIntegrityError(
                    "candidate reconciliation key does not match existing Problem provenance",
                )
            return problem_id

        return mint_problem_id()


def _build_resolved_problem(existing: Problem) -> Problem:
    return Problem(
        problem_id=existing.problem_id,
        tenant_id=existing.tenant_id,
        status=ProblemStatus.RESOLVED,
        first_seen_at=existing.first_seen_at,
        last_seen_at=existing.last_seen_at,
        occurrence_count=existing.occurrence_count,
        current_subject_refs=existing.current_subject_refs,
        occurrences=existing.occurrences,
        provenance=existing.provenance,
        record_version=existing.record_version + 1,
    )


def _validate_observed_at(observed_at: datetime) -> None:
    if type(observed_at) is not datetime:
        raise TypeError("observed_at must be datetime")
    if observed_at.tzinfo is None or observed_at.tzinfo.utcoffset(observed_at) is None:
        raise ValueError("observed_at must be timezone-aware")


def _validate_persisted_problem_identity(
    existing: Problem,
    *,
    tenant_id: str,
    problem_id: ProblemId,
    reconciliation_key: ProblemReconciliationKey,
) -> None:
    if existing.tenant_id != tenant_id:
        raise ProblemLifecycleIntegrityError(
            "tenant_id does not match Problem record during persistence reconciliation",
        )
    if existing.problem_id != problem_id:
        raise ProblemLifecycleIntegrityError(
            "problem identity changed during persistence reconciliation",
        )
    if not reconciliation_keys_equal(
        existing.provenance.reconciliation_key,
        reconciliation_key,
    ):
        raise ProblemLifecycleIntegrityError(
            "reconciliation identity mismatch during persistence reconciliation",
        )


def _validate_grouping_result_tenant(grouping_result: ProblemGroupingResult) -> None:
    for candidate in grouping_result.candidates:
        for member in candidate.members:
            if member.tenant_id != grouping_result.tenant_id:
                raise ProblemLifecycleIntegrityError(
                    "grouping result contains member outside invocation tenant scope",
                )


def _occurrence_for_member(
    *,
    member: ProblemGroupingSubjectRef,
    provenance: ProblemGroupingProvenance,
    observed_at: datetime,
) -> ProblemOccurrence:
    return ProblemOccurrence(
        subject_ref=member,
        observed_at=observed_at,
        strategy_id=provenance.strategy_id,
        strategy_version=provenance.strategy_version,
        method=provenance.method,
    )


def _create_problem(
    *,
    problem_id: ProblemId,
    tenant_id: str,
    candidate: ProblemGroupingCandidate,
    reconciliation_key: ProblemReconciliationKey,
    observed_at: datetime,
) -> Problem:
    occurrences = tuple(
        _occurrence_for_member(
            member=member,
            provenance=candidate.provenance,
            observed_at=observed_at,
        )
        for member in candidate.members
    )
    return Problem(
        problem_id=problem_id,
        tenant_id=tenant_id,
        status=ProblemStatus.OPEN,
        first_seen_at=observed_at,
        last_seen_at=observed_at,
        occurrence_count=len(occurrences),
        current_subject_refs=candidate.members,
        occurrences=occurrences,
        provenance=ProblemLifecycleProvenance(
            strategy_id=candidate.provenance.strategy_id,
            strategy_version=candidate.provenance.strategy_version,
            method=candidate.provenance.method,
            reconciliation_key=reconciliation_key,
        ),
        record_version=1,
    )


def _apply_candidate_to_problem(
    existing: Problem,
    *,
    candidate: ProblemGroupingCandidate,
    reconciliation_key: ProblemReconciliationKey,
    observed_at: datetime,
) -> tuple[Problem, bool]:
    if not reconciliation_keys_equal(
        existing.provenance.reconciliation_key,
        reconciliation_key,
    ):
        raise ProblemLifecycleIntegrityError(
            "candidate reconciliation key does not match existing Problem provenance",
        )

    known_subject_refs = {occurrence.subject_ref for occurrence in existing.occurrences}
    new_occurrences: list[ProblemOccurrence] = []
    for member in candidate.members:
        if member in known_subject_refs:
            continue
        new_occurrences.append(
            _occurrence_for_member(
                member=member,
                provenance=candidate.provenance,
                observed_at=observed_at,
            )
        )

    if not new_occurrences:
        return existing, False

    merged_occurrences = existing.occurrences + tuple(new_occurrences)
    merged_subject_refs = _merge_subject_refs(
        existing.current_subject_refs,
        candidate.members,
    )
    next_status = existing.status
    if existing.status is ProblemStatus.RESOLVED:
        next_status = ProblemStatus.OPEN

    next_record = Problem(
        problem_id=existing.problem_id,
        tenant_id=existing.tenant_id,
        status=next_status,
        first_seen_at=min(existing.first_seen_at, observed_at),
        last_seen_at=max(existing.last_seen_at, observed_at),
        occurrence_count=existing.occurrence_count + len(new_occurrences),
        current_subject_refs=merged_subject_refs,
        occurrences=merged_occurrences,
        provenance=ProblemLifecycleProvenance(
            strategy_id=candidate.provenance.strategy_id,
            strategy_version=candidate.provenance.strategy_version,
            method=candidate.provenance.method,
            reconciliation_key=reconciliation_key,
        ),
        record_version=existing.record_version + 1,
    )
    return next_record, True


def _merge_subject_refs(
    existing_refs: tuple[ProblemGroupingSubjectRef, ...],
    candidate_members: tuple[ProblemGroupingSubjectRef, ...],
) -> tuple[ProblemGroupingSubjectRef, ...]:
    merged: list[ProblemGroupingSubjectRef] = list(existing_refs)
    seen = set(existing_refs)
    for member in candidate_members:
        if member in seen:
            continue
        merged.append(member)
        seen.add(member)
    return tuple(merged)
