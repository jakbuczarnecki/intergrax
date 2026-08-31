# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory functional evidence persistence (tests, conformance, local lab)."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import DefaultDict

from intergrax.contracts.execution_identity import RunId, TaskId, validate_run_id, validate_task_id
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

_MAX_PAGE_SIZE = 1000


class InMemoryFunctionalEvidencePersistence(FunctionalEvidencePersistence):
    def __init__(self) -> None:
        self._accepted_by_evidence_id: dict[str, PlatformFunctionalEvidence] = {}
        self._by_execution: DefaultDict[tuple[str, str, str], list[str]] = defaultdict(list)
        self._lock = Lock()

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
            self._by_execution[execution_key].append(evidence.evidence_id)
            return evidence

    def query_evidence(self, request: FunctionalEvidenceQueryRequest) -> FunctionalEvidenceQueryPage:
        tenant_id = _require_tenant_id(request.tenant_id)
        task_id = validate_task_id(request.task_id)
        run_id = validate_run_id(request.run_id)
        page_size = _validate_page_size(request.page_size)
        execution_key = (tenant_id, task_id, run_id)
        with self._lock:
            records = self._resolve_ids(self._by_execution.get(execution_key, []))
            if request.kind is not None:
                records = tuple(record for record in records if record.kind is request.kind)
            start_index = _decode_cursor(request.cursor, tenant_id=tenant_id, task_id=task_id, run_id=run_id)
            page_items = records[start_index : start_index + page_size]
            next_index = start_index + len(page_items)
            next_cursor = _encode_cursor(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                index=next_index,
            ) if next_index < len(records) else None
            return FunctionalEvidenceQueryPage(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                items=page_items,
                next_cursor=next_cursor,
            )

    def _resolve_ids(self, evidence_ids: list[str]) -> tuple[PlatformFunctionalEvidence, ...]:
        records = [
            self._accepted_by_evidence_id[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in self._accepted_by_evidence_id
        ]
        records.sort(key=functional_evidence_query_order_key)
        return tuple(records)


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


def _encode_cursor(
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    index: int,
) -> str:
    return f"{tenant_id}|{task_id}|{run_id}|{index}"


def _decode_cursor(
    cursor: str | None,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
) -> int:
    if cursor is None:
        return 0
    if type(cursor) is not str or not cursor:
        raise FunctionalEvidencePersistenceIntegrityError("cursor must be a non-empty str")
    parts = cursor.split("|", 3)
    if len(parts) != 4:
        raise FunctionalEvidencePersistenceIntegrityError("cursor format invalid")
    cursor_tenant, cursor_task, cursor_run, index_text = parts
    if cursor_tenant != tenant_id or cursor_task != str(task_id) or cursor_run != str(run_id):
        raise FunctionalEvidencePersistenceIntegrityError("cursor scope mismatch")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise FunctionalEvidencePersistenceIntegrityError("cursor index invalid") from exc
    if index < 0:
        raise FunctionalEvidencePersistenceIntegrityError("cursor index must be >= 0")
    return index


__all__ = ["InMemoryFunctionalEvidencePersistence"]
