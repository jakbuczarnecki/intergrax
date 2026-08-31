# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded maintenance reconciliation for derived Problem list index projections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentQueryCursorCodec,
    DocumentRecord,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem
from intergrax.runtime.diagnostics.problem_list_query import (
    DecodedListIndexData,
    ProblemListScope,
    decode_list_index_data,
    encode_list_index_data,
    list_index_row_key,
    list_index_scope_from_row_key,
    problem_list_row_key_prefix,
)
from intergrax.runtime.diagnostics.problem_record_codec import decode_problem_record

_LIST_ROW_PREFIX = "list:"
_RECONCILE_PAGE_LIMIT = 500
_PROJECTION_HEALTH_DEGRADED_SKIP_THRESHOLD = 10

# Platform minimum destructive-maintenance age. Matches the 300s lease convention used
# across queueing/long-running workers and provides a conservative eventual-consistency
# window for derived list-index writes without requiring distributed writer leases.
MIN_SAFE_PROJECTION_AGE = timedelta(minutes=5)


class ProblemListIndexReconciliationError(ValueError):
    """Typed failure for invalid maintenance reconciliation parameters."""


class ProblemListIndexClassification(StrEnum):
    """Typed projection state for maintenance decisions."""

    CONSISTENT = "consistent"
    TRANSIENT_OR_UNCERTAIN = "transient_or_uncertain"
    PROVEN_STALE = "proven_stale"
    PROVEN_ORPHAN = "proven_orphan"
    CORRUPT = "corrupt"


class ProblemListProjectionHealth(StrEnum):
    """Process-local projection health derived from read and maintenance signals."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class ProblemListProjectionTelemetrySnapshot:
    """Immutable operator-visible counters for derived list projection quality."""

    skipped_missing_canonical: int
    skipped_version_ahead: int
    skipped_version_behind: int
    same_version_integrity_failure: int
    repaired_projection: int
    deleted_orphan_projection: int


@dataclass(slots=True)
class ProblemListProjectionTelemetry:
    """Mutable process-local projection telemetry (not RuntimeEvent evidence)."""

    skipped_missing_canonical: int = 0
    skipped_version_ahead: int = 0
    skipped_version_behind: int = 0
    same_version_integrity_failure: int = 0
    repaired_projection: int = 0
    deleted_orphan_projection: int = 0

    def snapshot(self) -> ProblemListProjectionTelemetrySnapshot:
        return ProblemListProjectionTelemetrySnapshot(
            skipped_missing_canonical=self.skipped_missing_canonical,
            skipped_version_ahead=self.skipped_version_ahead,
            skipped_version_behind=self.skipped_version_behind,
            same_version_integrity_failure=self.same_version_integrity_failure,
            repaired_projection=self.repaired_projection,
            deleted_orphan_projection=self.deleted_orphan_projection,
        )


@dataclass(frozen=True, slots=True)
class ProblemListIndexReconciliationPage:
    """Bounded maintenance page result for derived list index reconciliation."""

    examined: int
    consistent: int
    transient: int
    repaired: int
    deleted: int
    corrupt: int
    next_cursor: str | None
    has_more: bool


@dataclass(frozen=True, slots=True)
class ProblemListMaintenanceCycleKey:
    """Typed identity for one tenant/scope maintenance cycle."""

    tenant_id: str
    scope: ProblemListScope | None


@dataclass(slots=True)
class ProblemListMaintenanceCycleState:
    """Process-local maintenance-cycle continuity for projection health."""

    in_progress: bool = False
    had_issues: bool = False
    current_cycle_found_issues: bool = False
    started_at: datetime | None = None


@dataclass(slots=True)
class _ProblemListProjectionHealthState:
    last_query_skip_count: int = 0
    maintenance_cycles: dict[
        ProblemListMaintenanceCycleKey,
        ProblemListMaintenanceCycleState,
    ] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)


def resolve_effective_minimum_projection_age(
    *,
    minimum_projection_age: timedelta,
) -> timedelta:
    """Return caller-requested age clamped to the platform minimum (fail-closed)."""
    if minimum_projection_age < MIN_SAFE_PROJECTION_AGE:
        raise ProblemListIndexReconciliationError(
            "minimum_projection_age must be at least "
            f"{MIN_SAFE_PROJECTION_AGE.total_seconds():.0f} seconds",
        )
    return minimum_projection_age


def compute_safe_cutoff(
    *,
    now: datetime,
    minimum_projection_age: timedelta,
) -> datetime:
    """Derive destructive cutoff from reconciler clock authority."""
    if now.tzinfo is None:
        raise ProblemListIndexReconciliationError("reconciler clock must be timezone-aware UTC")
    return now - minimum_projection_age


def projection_age_is_below_destructive_threshold(
    *,
    projection_written_at: datetime | None,
    now: datetime,
    minimum_projection_age: timedelta,
) -> bool:
    """Return True when destructive maintenance must not run for this projection."""
    if projection_written_at is None:
        return True
    if projection_written_at.tzinfo is None:
        raise ProblemListIndexReconciliationError(
            "projection_written_at must be timezone-aware UTC",
        )
    if projection_written_at > now:
        return True
    age = now - projection_written_at
    return age <= minimum_projection_age


def classify_list_index_projection(
    *,
    index: DecodedListIndexData,
    canonical: Problem | None,
    now: datetime,
    minimum_projection_age: timedelta,
) -> ProblemListIndexClassification:
    """
    Classify one derived list index row against canonical truth.

    Destructive classifications require ``projection_written_at`` to be strictly older
    than ``minimum_projection_age`` relative to the reconciler clock. Future-dated
    projections and v1 rows without timestamps remain ``TRANSIENT_OR_UNCERTAIN``.
    """
    too_young = projection_age_is_below_destructive_threshold(
        projection_written_at=index.projection_written_at,
        now=now,
        minimum_projection_age=minimum_projection_age,
    )

    if canonical is None:
        if too_young:
            return ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN
        return ProblemListIndexClassification.PROVEN_ORPHAN

    if canonical.problem_id != index.problem_id:
        return ProblemListIndexClassification.CORRUPT

    if index.record_version == canonical.record_version:
        if (
            index.last_seen_at == canonical.last_seen_at
            and index.status is canonical.status
        ):
            return ProblemListIndexClassification.CONSISTENT
        return ProblemListIndexClassification.CORRUPT

    if too_young:
        return ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN

    return ProblemListIndexClassification.PROVEN_STALE


def maintenance_cycle_degrades_health(
    *,
    cycle_state: ProblemListMaintenanceCycleState,
) -> bool:
    """Return True when a tracked cycle still blocks HEALTHY recovery."""
    return cycle_state.had_issues


def projection_health_from_state(
    *,
    telemetry: ProblemListProjectionTelemetry,
    health_state: _ProblemListProjectionHealthState,
) -> ProblemListProjectionHealth:
    if telemetry.same_version_integrity_failure > 0:
        return ProblemListProjectionHealth.DEGRADED
    with health_state._lock:
        for cycle_state in health_state.maintenance_cycles.values():
            if maintenance_cycle_degrades_health(cycle_state=cycle_state):
                return ProblemListProjectionHealth.DEGRADED
    if health_state.last_query_skip_count > _PROJECTION_HEALTH_DEGRADED_SKIP_THRESHOLD:
        return ProblemListProjectionHealth.DEGRADED
    return ProblemListProjectionHealth.HEALTHY


@dataclass
class ProblemListIndexReconciler:
    """Bounded maintenance reconciler for derived Problem list index projections."""

    document_store: ConditionalDocumentStore
    document_query_cursor_codec: DocumentQueryCursorCodec
    clock: Callable[[], datetime]
    telemetry: ProblemListProjectionTelemetry = field(
        default_factory=ProblemListProjectionTelemetry,
    )
    health_state: _ProblemListProjectionHealthState = field(
        default_factory=_ProblemListProjectionHealthState,
    )

    def reconcile_list_indexes(
        self,
        *,
        tenant_id: str,
        minimum_projection_age: timedelta = MIN_SAFE_PROJECTION_AGE,
        scope: ProblemListScope | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> ProblemListIndexReconciliationPage:
        if type(limit) is not int or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive int")

        effective_minimum_age = resolve_effective_minimum_projection_age(
            minimum_projection_age=minimum_projection_age,
        )
        now = self.clock()
        cycle_key = ProblemListMaintenanceCycleKey(
            tenant_id=tenant_id,
            scope=scope,
        )
        cycle_state = self._begin_or_continue_cycle(
            cycle_key=cycle_key,
            cursor=cursor,
            now=now,
        )

        partition_key = f"intergrax.diagnostic_problem.v1:{tenant_id}"
        row_key_prefix = (
            problem_list_row_key_prefix(scope)
            if scope is not None
            else _LIST_ROW_PREFIX
        )
        store_cursor: str | None = cursor
        if cursor is not None:
            self.document_query_cursor_codec.decode(
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                cursor=cursor,
            )

        examined = 0
        consistent = 0
        transient = 0
        repaired = 0
        deleted = 0
        corrupt = 0
        last_row_key: str | None = None
        continuation = store_cursor
        has_more = False

        while examined < limit:
            fetch_limit = min(limit - examined, _RECONCILE_PAGE_LIMIT)
            page = self.document_store.query(
                partition_key,
                limit=fetch_limit,
                row_key_prefix=row_key_prefix,
                cursor=continuation,
            )
            if not page.documents:
                break

            for index_document in page.documents:
                if examined >= limit:
                    has_more = True
                    break
                examined += 1
                last_row_key = index_document.row_key
                outcome = self._reconcile_one(
                    index_document=index_document,
                    tenant_id=tenant_id,
                    partition_key=partition_key,
                    now=now,
                    minimum_projection_age=effective_minimum_age,
                )
                if outcome is ProblemListIndexClassification.CONSISTENT:
                    consistent += 1
                elif outcome is ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN:
                    transient += 1
                elif outcome is ProblemListIndexClassification.PROVEN_STALE:
                    repaired += 1
                elif outcome is ProblemListIndexClassification.PROVEN_ORPHAN:
                    deleted += 1
                else:
                    corrupt += 1

            if examined >= limit:
                has_more = (
                    page.next_cursor is not None
                    or len(page.documents) >= fetch_limit
                )
                break

            if page.next_cursor is None:
                break
            continuation = page.next_cursor

        next_cursor: str | None = None
        if has_more and last_row_key is not None:
            next_cursor = self.document_query_cursor_codec.encode(
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                last_row_key=last_row_key,
            )

        if repaired > 0 or deleted > 0 or corrupt > 0:
            with self.health_state._lock:
                cycle_state.current_cycle_found_issues = True
                cycle_state.had_issues = True

        if not has_more:
            self._complete_maintenance_cycle(
                cycle_key=cycle_key,
                cycle_state=cycle_state,
            )

        return ProblemListIndexReconciliationPage(
            examined=examined,
            consistent=consistent,
            transient=transient,
            repaired=repaired,
            deleted=deleted,
            corrupt=corrupt,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    def _begin_or_continue_cycle(
        self,
        *,
        cycle_key: ProblemListMaintenanceCycleKey,
        cursor: str | None,
        now: datetime,
    ) -> ProblemListMaintenanceCycleState:
        with self.health_state._lock:
            cycles = self.health_state.maintenance_cycles
            existing = cycles.get(cycle_key)
            if cursor is None:
                if existing is not None and existing.in_progress:
                    raise ProblemListIndexReconciliationError(
                        "maintenance cycle already in progress for tenant/scope; "
                        "continuation cursor required",
                    )
                cycle_state = ProblemListMaintenanceCycleState(
                    in_progress=True,
                    had_issues=existing.had_issues if existing is not None else False,
                    current_cycle_found_issues=False,
                    started_at=now,
                )
                cycles[cycle_key] = cycle_state
                return cycle_state

            if existing is None:
                cycle_state = ProblemListMaintenanceCycleState(
                    in_progress=True,
                    had_issues=False,
                    current_cycle_found_issues=False,
                    started_at=now,
                )
                cycles[cycle_key] = cycle_state
                return cycle_state
            if not existing.in_progress:
                raise ProblemListIndexReconciliationError(
                    "maintenance continuation cursor does not match an in-progress cycle",
                )
            return existing

    def _complete_maintenance_cycle(
        self,
        *,
        cycle_key: ProblemListMaintenanceCycleKey,
        cycle_state: ProblemListMaintenanceCycleState,
    ) -> None:
        with self.health_state._lock:
            cycle_state.in_progress = False
            if (
                not cycle_state.had_issues
                or (
                    cycle_state.had_issues
                    and not cycle_state.current_cycle_found_issues
                )
            ):
                self.health_state.maintenance_cycles.pop(cycle_key, None)

    def _reconcile_one(
        self,
        *,
        index_document: DocumentRecord,
        tenant_id: str,
        partition_key: str,
        now: datetime,
        minimum_projection_age: timedelta,
    ) -> ProblemListIndexClassification:
        try:
            index = decode_list_index_data(dict(index_document.data))
        except ValueError:
            return ProblemListIndexClassification.CORRUPT

        scope = list_index_scope_from_row_key(index_document.row_key)
        if scope is None:
            return ProblemListIndexClassification.CORRUPT

        canonical_record = self.document_store.get(
            partition_key,
            f"record:{index.problem_id}",
        )
        canonical: Problem | None = None
        if canonical_record is not None:
            canonical = decode_problem_record(dict(canonical_record.data))
            if canonical.tenant_id != tenant_id or canonical.problem_id != index.problem_id:
                return ProblemListIndexClassification.CORRUPT

        classification = classify_list_index_projection(
            index=index,
            canonical=canonical,
            now=now,
            minimum_projection_age=minimum_projection_age,
        )

        if classification is ProblemListIndexClassification.CONSISTENT:
            if index.projection_written_at is None and canonical is not None:
                upgraded = self._build_index_document(
                    canonical=canonical,
                    scope=scope,
                    partition_key=partition_key,
                    projection_written_at=self.clock(),
                )
                if self.document_store.replace_if_match(
                    expected=index_document,
                    replacement=upgraded,
                ):
                    self.telemetry.repaired_projection += 1
                    return ProblemListIndexClassification.PROVEN_STALE
            return ProblemListIndexClassification.CONSISTENT

        if classification is ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN:
            return classification

        if classification is ProblemListIndexClassification.CORRUPT:
            return classification

        if classification is ProblemListIndexClassification.PROVEN_ORPHAN:
            if self.document_store.delete_if_match(expected=index_document):
                self.telemetry.deleted_orphan_projection += 1
                return ProblemListIndexClassification.PROVEN_ORPHAN
            return ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN

        if canonical is None:
            return ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN

        repaired_document = self._build_index_document(
            canonical=canonical,
            scope=scope,
            partition_key=partition_key,
            projection_written_at=self.clock(),
        )
        if repaired_document.row_key == index_document.row_key:
            if self.document_store.replace_if_match(
                expected=index_document,
                replacement=repaired_document,
            ):
                self.telemetry.repaired_projection += 1
                return ProblemListIndexClassification.PROVEN_STALE
            return ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN

        if not self.document_store.put_if_absent(repaired_document):
            existing = self.document_store.get(
                repaired_document.partition_key,
                repaired_document.row_key,
            )
            if existing is None or dict(existing.data) != dict(repaired_document.data):
                return ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN
        if self.document_store.delete_if_match(expected=index_document):
            self.telemetry.repaired_projection += 1
            return ProblemListIndexClassification.PROVEN_STALE
        return ProblemListIndexClassification.TRANSIENT_OR_UNCERTAIN

    @staticmethod
    def _build_index_document(
        *,
        canonical: Problem,
        scope: ProblemListScope,
        partition_key: str,
        projection_written_at: datetime,
    ) -> DocumentRecord:
        if projection_written_at.tzinfo is None:
            projection_written_at = projection_written_at.replace(tzinfo=UTC)
        return DocumentRecord(
            partition_key=partition_key,
            row_key=list_index_row_key(scope=scope, problem=canonical),
            data=encode_list_index_data(
                problem_id=canonical.problem_id,
                last_seen_at=canonical.last_seen_at,
                status=canonical.status,
                record_version=canonical.record_version,
                projection_written_at=projection_written_at,
            ),
        )
