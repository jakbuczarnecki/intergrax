# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Derived execution lineage reconstruction helpers (DG-001 read integration R1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, TypeVar

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionRecord,
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineageIntegrityError,
    ExecutionLineageReader,
    ExecutionLineageSegmentLifecycle,
    ExecutionLineageSegmentRecord,
    ExecutionLineageSealRecord,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)


_TPage = TypeVar("_TPage")
_TItem = TypeVar("_TItem")


class ExecutionLineageReconstructionIntegrityError(Exception):
    """Raised when durable lineage fails forensic structural validation."""


class ExecutionLineageReadStatus(StrEnum):
    AVAILABLE = "available"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"


class ExecutionLineageCompleteness(StrEnum):
    OPEN = "open"
    COMPLETE = "complete"
    PARTIAL = "partial"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class ReconstructedLineageSegment:
    root_execution_id: ExecutionId
    predecessor_root_execution_id: ExecutionId | None
    lifecycle: ExecutionLineageSegmentLifecycle
    admissions: tuple[ExecutionLineageAdmissionRecord, ...]


@dataclass(frozen=True, slots=True)
class ReconstructedAttemptLineage:
    attempt_id: AttemptId
    read_status: ExecutionLineageReadStatus
    completeness: ExecutionLineageCompleteness | None
    degraded: bool | None
    closure_kind: ExecutionLineageAttemptClosureKind | None
    segments: tuple[ReconstructedLineageSegment, ...]


def reconstruct_attempt_lineage(
    reader: ExecutionLineageReader,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    initial_lineage_page_limit: int,
    max_lineage_records: int,
) -> ReconstructedAttemptLineage:
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    try:
        attempt_state = reader.read_attempt_lineage_state(scope)
    except ExecutionLineageUnavailableError:
        return ReconstructedAttemptLineage(
            attempt_id=attempt_id,
            read_status=ExecutionLineageReadStatus.UNAVAILABLE,
            completeness=None,
            degraded=None,
            closure_kind=None,
            segments=(),
        )
    if attempt_state is None:
        return ReconstructedAttemptLineage(
            attempt_id=attempt_id,
            read_status=ExecutionLineageReadStatus.ABSENT,
            completeness=None,
            degraded=None,
            closure_kind=None,
            segments=(),
        )

    try:
        segments_raw, segments_truncated = _load_all_segments(
            reader,
            scope,
            page_limit=initial_lineage_page_limit,
            max_records=max_lineage_records,
        )
        admissions_raw, admissions_truncated = _load_all_admissions(
            reader,
            scope,
            page_limit=initial_lineage_page_limit,
            max_records=max_lineage_records,
        )
        seal = reader.read_seal(scope)
    except ExecutionLineageUnavailableError:
        return ReconstructedAttemptLineage(
            attempt_id=attempt_id,
            read_status=ExecutionLineageReadStatus.UNAVAILABLE,
            completeness=None,
            degraded=None,
            closure_kind=None,
            segments=(),
        )
    except ExecutionLineageIntegrityError as exc:
        raise ExecutionLineageReconstructionIntegrityError(str(exc)) from exc

    _validate_lineage_scope(
        scope,
        attempt_state=attempt_state,
        segments=segments_raw,
        admissions=admissions_raw,
        seal=seal,
    )
    _validate_execution_id_uniqueness(admissions_raw)
    _validate_segment_roots(segments_raw, admissions_raw)
    _validate_admission_segment_membership(segments_raw, admissions_raw)
    _validate_parent_edges(segments_raw, admissions_raw)
    ordered_segments = _order_segments_by_continuity(segments_raw)
    _validate_segment_continuity(ordered_segments)
    segments = _build_reconstructed_segments(ordered_segments, admissions_raw)

    truncated = segments_truncated or admissions_truncated
    completeness = _derive_completeness(
        attempt_state=attempt_state,
        seal=seal,
        segments=segments_raw,
        truncated=truncated,
    )
    closure_kind = attempt_state.closure_kind
    if seal is not None:
        closure_kind = seal.closure_kind

    return ReconstructedAttemptLineage(
        attempt_id=attempt_id,
        read_status=ExecutionLineageReadStatus.AVAILABLE,
        completeness=completeness,
        degraded=attempt_state.degraded,
        closure_kind=closure_kind,
        segments=segments,
    )


def _load_all_admissions(
    reader: ExecutionLineageReader,
    scope: ExecutionLineageAttemptScope,
    *,
    page_limit: int,
    max_records: int,
) -> tuple[tuple[ExecutionLineageAdmissionRecord, ...], bool]:
    return _load_bounded_pages(
        load_page=lambda cursor: reader.list_admissions_for_attempt(
            scope,
            page_limit,
            cursor=cursor,
        ),
        items=lambda page: page.admissions,
        next_cursor=lambda page: page.next_cursor,
        max_records=max_records,
    )


def _load_all_segments(
    reader: ExecutionLineageReader,
    scope: ExecutionLineageAttemptScope,
    *,
    page_limit: int,
    max_records: int,
) -> tuple[tuple[ExecutionLineageSegmentRecord, ...], bool]:
    return _load_bounded_pages(
        load_page=lambda cursor: reader.list_segments_for_attempt(
            scope,
            page_limit,
            cursor=cursor,
        ),
        items=lambda page: page.segments,
        next_cursor=lambda page: page.next_cursor,
        max_records=max_records,
    )


def _load_bounded_pages(
    *,
    load_page: Callable[[str | None], _TPage],
    items: Callable[[_TPage], tuple[_TItem, ...]],
    next_cursor: Callable[[_TPage], str | None],
    max_records: int,
) -> tuple[tuple[_TItem, ...], bool]:
    collected: list[_TItem] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    truncated = False
    while True:
        if cursor is not None:
            if cursor in seen_cursors:
                raise ExecutionLineageReconstructionIntegrityError(
                    "lineage cursor cycle"
                )
            seen_cursors.add(cursor)
        page = load_page(cursor)
        batch = items(page)
        collected.extend(batch)
        if len(collected) > max_records:
            truncated = True
            collected = collected[:max_records]
            break
        page_next = next_cursor(page)
        if page_next is None:
            break
        if page_next == cursor:
            raise ExecutionLineageReconstructionIntegrityError("lineage cursor cycle")
        cursor = page_next
    return tuple(collected), truncated


def _validate_lineage_scope(
    scope: ExecutionLineageAttemptScope,
    *,
    attempt_state: ExecutionLineageAttemptState,
    segments: tuple[ExecutionLineageSegmentRecord, ...],
    admissions: tuple[ExecutionLineageAdmissionRecord, ...],
    seal: ExecutionLineageSealRecord | None,
) -> None:
    if not _scopes_match(scope, attempt_state.scope):
        raise ExecutionLineageReconstructionIntegrityError(
            "attempt state scope mismatch"
        )
    for segment in segments:
        if not _scopes_match(scope, segment.scope):
            raise ExecutionLineageReconstructionIntegrityError("segment scope mismatch")
    for admission in admissions:
        if not _scopes_match(scope, admission.scope):
            raise ExecutionLineageReconstructionIntegrityError(
                "admission scope mismatch"
            )
    if seal is not None and not _scopes_match(scope, seal.scope):
        raise ExecutionLineageReconstructionIntegrityError("seal scope mismatch")


def _validate_execution_id_uniqueness(
    admissions: tuple[ExecutionLineageAdmissionRecord, ...],
) -> None:
    seen: set[ExecutionId] = set()
    for admission in admissions:
        if admission.execution_id in seen:
            raise ExecutionLineageReconstructionIntegrityError(
                "duplicate durable execution admission",
            )
        seen.add(admission.execution_id)


def _validate_segment_roots(
    segments: tuple[ExecutionLineageSegmentRecord, ...],
    admissions: tuple[ExecutionLineageAdmissionRecord, ...],
) -> None:
    admissions_by_segment: dict[ExecutionId, list[ExecutionLineageAdmissionRecord]] = {}
    for admission in admissions:
        admissions_by_segment.setdefault(
            admission.segment_root_execution_id,
            [],
        ).append(admission)

    for segment in segments:
        segment_admissions = admissions_by_segment.get(segment.root_execution_id, ())
        roots = [
            admission
            for admission in segment_admissions
            if admission.parent_execution_id is None
            and admission.execution_id == segment.root_execution_id
            and admission.segment_root_execution_id == segment.root_execution_id
        ]
        if len(roots) != 1:
            raise ExecutionLineageReconstructionIntegrityError(
                "segment root admission invalid",
            )


def _validate_admission_segment_membership(
    segments: tuple[ExecutionLineageSegmentRecord, ...],
    admissions: tuple[ExecutionLineageAdmissionRecord, ...],
) -> None:
    segment_roots = {segment.root_execution_id for segment in segments}
    for admission in admissions:
        if admission.segment_root_execution_id not in segment_roots:
            raise ExecutionLineageReconstructionIntegrityError(
                "admission references unknown segment",
            )


def _validate_parent_edges(
    segments: tuple[ExecutionLineageSegmentRecord, ...],
    admissions: tuple[ExecutionLineageAdmissionRecord, ...],
) -> None:
    admissions_by_segment: dict[
        ExecutionId, dict[ExecutionId, ExecutionLineageAdmissionRecord]
    ] = {}
    for admission in admissions:
        segment_map = admissions_by_segment.setdefault(
            admission.segment_root_execution_id,
            {},
        )
        segment_map[admission.execution_id] = admission

    for segment in segments:
        segment_map = admissions_by_segment.get(segment.root_execution_id, {})
        for admission in segment_map.values():
            parent_id = admission.parent_execution_id
            if parent_id is None:
                continue
            if parent_id not in segment_map:
                raise ExecutionLineageReconstructionIntegrityError(
                    "child admission parent missing in segment history",
                )


def _order_segments_by_continuity(
    segments: tuple[ExecutionLineageSegmentRecord, ...],
) -> tuple[ExecutionLineageSegmentRecord, ...]:
    by_root = {segment.root_execution_id: segment for segment in segments}
    initial = [
        segment for segment in segments if segment.predecessor_root_execution_id is None
    ]
    if len(initial) != 1:
        raise ExecutionLineageReconstructionIntegrityError(
            "segment continuity requires exactly one initial segment",
        )
    ordered: list[ExecutionLineageSegmentRecord] = []
    current: ExecutionLineageSegmentRecord | None = initial[0]
    visited: set[ExecutionId] = set()
    while current is not None:
        if current.root_execution_id in visited:
            raise ExecutionLineageReconstructionIntegrityError(
                "segment continuity cycle"
            )
        visited.add(current.root_execution_id)
        ordered.append(current)
        successor = _find_segment_successor(current.root_execution_id, by_root)
        current = successor
    if len(visited) != len(segments):
        raise ExecutionLineageReconstructionIntegrityError(
            "segment predecessor missing"
        )
    return tuple(ordered)


def _find_segment_successor(
    predecessor_root: ExecutionId,
    by_root: dict[ExecutionId, ExecutionLineageSegmentRecord],
) -> ExecutionLineageSegmentRecord | None:
    matches = [
        segment
        for segment in by_root.values()
        if segment.predecessor_root_execution_id == predecessor_root
    ]
    if len(matches) > 1:
        raise ExecutionLineageReconstructionIntegrityError(
            "segment continuity allows at most one successor",
        )
    return matches[0] if matches else None


def _validate_segment_continuity(
    ordered_segments: tuple[ExecutionLineageSegmentRecord, ...],
) -> None:
    for index, segment in enumerate(ordered_segments):
        if index == 0:
            if segment.predecessor_root_execution_id is not None:
                raise ExecutionLineageReconstructionIntegrityError(
                    "initial segment must not have predecessor",
                )
            continue
        predecessor = segment.predecessor_root_execution_id
        if predecessor is None:
            raise ExecutionLineageReconstructionIntegrityError(
                "segment predecessor missing"
            )
        if predecessor == segment.root_execution_id:
            raise ExecutionLineageReconstructionIntegrityError(
                "segment continuity cycle"
            )
        previous = ordered_segments[index - 1]
        if predecessor != previous.root_execution_id:
            raise ExecutionLineageReconstructionIntegrityError(
                "segment continuity chain broken",
            )


def _build_reconstructed_segments(
    ordered_segments: tuple[ExecutionLineageSegmentRecord, ...],
    admissions: tuple[ExecutionLineageAdmissionRecord, ...],
) -> tuple[ReconstructedLineageSegment, ...]:
    admissions_by_segment: dict[ExecutionId, list[ExecutionLineageAdmissionRecord]] = {}
    for admission in admissions:
        admissions_by_segment.setdefault(
            admission.segment_root_execution_id,
            [],
        ).append(admission)

    reconstructed: list[ReconstructedLineageSegment] = []
    for segment in ordered_segments:
        segment_admissions = tuple(
            sorted(
                admissions_by_segment.get(segment.root_execution_id, ()),
                key=lambda item: item.admission_position,
            ),
        )
        reconstructed.append(
            ReconstructedLineageSegment(
                root_execution_id=segment.root_execution_id,
                predecessor_root_execution_id=segment.predecessor_root_execution_id,
                lifecycle=segment.lifecycle,
                admissions=segment_admissions,
            ),
        )
    return tuple(reconstructed)


def _derive_completeness(
    *,
    attempt_state: ExecutionLineageAttemptState,
    seal: ExecutionLineageSealRecord | None,
    segments: tuple[ExecutionLineageSegmentRecord, ...],
    truncated: bool,
) -> ExecutionLineageCompleteness:
    if truncated:
        return ExecutionLineageCompleteness.TRUNCATED
    if attempt_state.degraded:
        return ExecutionLineageCompleteness.PARTIAL
    if any(
        segment.lifecycle is ExecutionLineageSegmentLifecycle.SEGMENT_UNCLEAN
        for segment in segments
    ):
        return ExecutionLineageCompleteness.PARTIAL
    if not attempt_state.sealed:
        return ExecutionLineageCompleteness.OPEN
    if seal is None or attempt_state.degraded:
        return ExecutionLineageCompleteness.PARTIAL
    return ExecutionLineageCompleteness.COMPLETE


def _scopes_match(
    left: ExecutionLineageAttemptScope,
    right: ExecutionLineageAttemptScope,
) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.task_id == right.task_id
        and left.run_id == right.run_id
        and left.attempt_id == right.attempt_id
    )


__all__ = [
    "ExecutionLineageCompleteness",
    "ExecutionLineageReadStatus",
    "ExecutionLineageReconstructionIntegrityError",
    "ReconstructedAttemptLineage",
    "ReconstructedLineageSegment",
    "reconstruct_attempt_lineage",
]
