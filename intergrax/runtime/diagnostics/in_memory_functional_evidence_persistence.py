# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory functional evidence persistence (tests, conformance, local lab)."""

from __future__ import annotations

import bisect
import secrets
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import DefaultDict

from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.diagnostics.functional_evidence import (
    PipelineEvidenceKind,
    PlatformFunctionalEvidence,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceConflictError,
    FunctionalEvidencePersistenceIntegrityError,
    FunctionalEvidenceQueryPage,
    FunctionalEvidenceQueryRequest,
    functional_evidence_query_order_key,
)
from intergrax.runtime.diagnostics.functional_evidence_query_cursor import (
    FunctionalEvidenceQueryCursorCodec,
    FunctionalEvidenceQueryCursorError,
)

_MAX_PAGE_SIZE = 1000
_DEFAULT_CURSOR_SECRET = secrets.token_bytes(32)


@dataclass(frozen=True, order=True, slots=True)
class _SortedEntry:
    order_key: tuple[datetime, str]
    evidence_id: str


class InMemoryFunctionalEvidencePersistence(FunctionalEvidencePersistence):
    def __init__(
        self,
        *,
        cursor_codec: FunctionalEvidenceQueryCursorCodec | None = None,
        cursor_secret: bytes | None = None,
    ) -> None:
        if cursor_codec is not None and cursor_secret is not None:
            raise TypeError("functional_evidence_cursor_codec_configuration_invalid")
        if cursor_secret is not None:
            self._cursor_codec = FunctionalEvidenceQueryCursorCodec(secret=cursor_secret)
        else:
            self._cursor_codec = cursor_codec or FunctionalEvidenceQueryCursorCodec(
                secret=_DEFAULT_CURSOR_SECRET,
            )
        self._accepted_by_evidence_id: dict[str, PlatformFunctionalEvidence] = {}
        self._sorted_by_execution: DefaultDict[tuple[str, str, str], list[_SortedEntry]] = (
            defaultdict(list)
        )
        self._lock = Lock()

    @property
    def query_cursor_codec(self) -> FunctionalEvidenceQueryCursorCodec:
        return self._cursor_codec

    def append(self, evidence: PlatformFunctionalEvidence) -> PlatformFunctionalEvidence:
        with self._lock:
            existing = self._accepted_by_evidence_id.get(evidence.evidence_id)
            if existing is not None:
                if existing != evidence:
                    raise FunctionalEvidencePersistenceConflictError(
                        "conflicting functional evidence for evidence_id",
                    )
                return existing
            self._accepted_by_evidence_id[evidence.evidence_id] = evidence
            execution_key = (
                evidence.scope.tenant_id,
                evidence.scope.task_id,
                evidence.scope.run_id,
            )
            entry = _SortedEntry(
                order_key=functional_evidence_query_order_key(evidence),
                evidence_id=evidence.evidence_id,
            )
            bisect.insort(self._sorted_by_execution[execution_key], entry)
            return evidence

    def query_evidence(self, request: FunctionalEvidenceQueryRequest) -> FunctionalEvidenceQueryPage:
        tenant_id = _require_tenant_id(request.tenant_id)
        task_id = validate_task_id(request.task_id)
        run_id = validate_run_id(request.run_id)
        attempt_id = _validate_attempt_filter(request.attempt_id)
        page_size = _validate_page_size(request.page_size)
        execution_key = (tenant_id, task_id, run_id)
        with self._lock:
            sorted_entries = self._sorted_by_execution.get(execution_key, [])
            start_pos = _resolve_start_position(
                sorted_entries=sorted_entries,
                cursor=request.cursor,
                cursor_codec=self._cursor_codec,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                kind=request.kind,
            )
            items, scan_pos = _collect_page(
                sorted_entries=sorted_entries,
                records_by_id=self._accepted_by_evidence_id,
                start_pos=start_pos,
                page_size=page_size,
                attempt_id=attempt_id,
                kind=request.kind,
            )
            next_cursor = _resolve_next_cursor(
                sorted_entries=sorted_entries,
                records_by_id=self._accepted_by_evidence_id,
                items=items,
                scan_pos=scan_pos,
                cursor_codec=self._cursor_codec,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                kind=request.kind,
            )
            return FunctionalEvidenceQueryPage(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                items=items,
                next_cursor=next_cursor,
            )


def _resolve_start_position(
    *,
    sorted_entries: list[_SortedEntry],
    cursor: str | None,
    cursor_codec: FunctionalEvidenceQueryCursorCodec,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> int:
    if cursor is None:
        return 0
    try:
        payload = cursor_codec.decode(
            cursor,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            kind=kind,
        )
    except FunctionalEvidenceQueryCursorError as exc:
        raise FunctionalEvidencePersistenceIntegrityError(str(exc)) from exc
    cursor_key = (payload.last_recorded_at, payload.last_evidence_id)
    return bisect.bisect_right(
        sorted_entries,
        _SortedEntry(order_key=cursor_key, evidence_id=payload.last_evidence_id),
    )


def _matches_filters(
    record: PlatformFunctionalEvidence,
    *,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> bool:
    if kind is not None and record.kind is not kind:
        return False
    if attempt_id is not None and record.scope.attempt_id != attempt_id:
        return False
    return True


def _collect_page(
    *,
    sorted_entries: list[_SortedEntry],
    records_by_id: dict[str, PlatformFunctionalEvidence],
    start_pos: int,
    page_size: int,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> tuple[tuple[PlatformFunctionalEvidence, ...], int]:
    items: list[PlatformFunctionalEvidence] = []
    scan_pos = start_pos
    while scan_pos < len(sorted_entries) and len(items) < page_size:
        entry = sorted_entries[scan_pos]
        scan_pos += 1
        record = records_by_id[entry.evidence_id]
        if not _matches_filters(record, attempt_id=attempt_id, kind=kind):
            continue
        items.append(record)
    return tuple(items), scan_pos


def _resolve_next_cursor(
    *,
    sorted_entries: list[_SortedEntry],
    records_by_id: dict[str, PlatformFunctionalEvidence],
    items: tuple[PlatformFunctionalEvidence, ...],
    scan_pos: int,
    cursor_codec: FunctionalEvidenceQueryCursorCodec,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> str | None:
    if not items:
        return None
    if _has_more_matching(
        sorted_entries=sorted_entries,
        records_by_id=records_by_id,
        start_pos=scan_pos,
        attempt_id=attempt_id,
        kind=kind,
    ):
        last_item = items[-1]
        return cursor_codec.encode(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            kind=kind,
            last_recorded_at=last_item.provenance.recorded_at,
            last_evidence_id=last_item.evidence_id,
        )
    return None


def _has_more_matching(
    *,
    sorted_entries: list[_SortedEntry],
    records_by_id: dict[str, PlatformFunctionalEvidence],
    start_pos: int,
    attempt_id: AttemptId | None,
    kind: PipelineEvidenceKind | None,
) -> bool:
    scan_pos = start_pos
    while scan_pos < len(sorted_entries):
        entry = sorted_entries[scan_pos]
        scan_pos += 1
        record = records_by_id[entry.evidence_id]
        if _matches_filters(record, attempt_id=attempt_id, kind=kind):
            return True
    return False


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise FunctionalEvidencePersistenceIntegrityError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise FunctionalEvidencePersistenceIntegrityError("tenant_id must be non-empty")
    if tenant_id != normalized:
        raise FunctionalEvidencePersistenceIntegrityError(
            "tenant_id must not contain leading or trailing whitespace",
        )
    return normalized


def _validate_attempt_filter(attempt_id: AttemptId | None) -> AttemptId | None:
    if attempt_id is None:
        return None
    return validate_attempt_id(attempt_id)


def _validate_page_size(page_size: int) -> int:
    if type(page_size) is not int or isinstance(page_size, bool):
        raise FunctionalEvidencePersistenceIntegrityError("page_size must be int")
    if page_size < 1:
        raise FunctionalEvidencePersistenceIntegrityError("page_size must be >= 1")
    if page_size > _MAX_PAGE_SIZE:
        raise FunctionalEvidencePersistenceIntegrityError(
            f"page_size must be <= {_MAX_PAGE_SIZE}",
        )
    return page_size


__all__ = ["InMemoryFunctionalEvidencePersistence"]
