# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Legacy inline occurrence migration to durable occurrence persistence (DIAG-ENTERPRISE-2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore, DocumentRecord
from intergrax.runtime.diagnostics.document_store_problem_persistence import (
    DocumentStoreProblemPersistence,
    _document_partition,
    _record_row_key,
)
from intergrax.runtime.diagnostics.problem_lifecycle import Problem, ProblemId
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.problem_persistence import ProblemPersistence
from intergrax.runtime.diagnostics.problem_record_codec import (
    decode_legacy_problem_record_with_occurrences,
    decode_problem_record,
    encode_problem_record,
)


@dataclass(frozen=True, slots=True)
class ProblemOccurrenceMigrationPage:
    migrated_problem_ids: tuple[ProblemId, ...]
    next_cursor: str | None
    has_more: bool


def migrate_legacy_problem_inline_occurrences(
    *,
    tenant_id: str,
    problem_persistence: ProblemPersistence,
    occurrence_persistence: ProblemOccurrencePersistence,
    document_store: ConditionalDocumentStore,
    limit: int = 10,
    cursor: str | None = None,
) -> ProblemOccurrenceMigrationPage:
    """
    Bounded v1→v2 migration: inline occurrences become durable rows; Problem rewritten v2.

    Idempotent and crash-safe: occurrence append uses ``append_if_absent``.
    """
    if type(limit) is not int or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive int")

    partition_key = _document_partition(tenant_id)
    page = document_store.query(
        partition_key,
        limit=limit,
        row_key_prefix="record:",
        cursor=cursor,
    )
    migrated: list[ProblemId] = []
    for document in page.documents:
        raw = dict(document.data)
        schema_version = raw.get("schema_version")
        if schema_version != "intergrax.diagnostic_problem.persistence.v1":
            continue
        problem, inline_occurrences, inline_subject_refs = (
            decode_legacy_problem_record_with_occurrences(raw)
        )
        if problem.tenant_id != tenant_id:
            continue
        for occurrence in inline_occurrences:
            occurrence_persistence.append_if_absent(
                tenant_id=tenant_id,
                problem_id=problem.problem_id,
                occurrence=occurrence,
            )
        bounded = Problem(
            problem_id=problem.problem_id,
            tenant_id=problem.tenant_id,
            status=problem.status,
            first_seen_at=problem.first_seen_at,
            last_seen_at=problem.last_seen_at,
            occurrence_count=problem.occurrence_count,
            provenance=problem.provenance,
            record_version=problem.record_version,
        )
        replacement = DocumentRecord(
            partition_key=partition_key,
            row_key=_record_row_key(problem.problem_id),
            data=encode_problem_record(bounded),
        )
        document_store.replace_if_match(expected=document, replacement=replacement)

        existing = problem_persistence.get(
            tenant_id=tenant_id,
            problem_id=problem.problem_id,
        )
        if existing is None:
            problem_persistence.create(bounded, indexed_subject_refs=())

        if isinstance(problem_persistence, DocumentStoreProblemPersistence):
            problem_persistence._ensure_reconciliation_index(
                record=bounded,
                partition_key=partition_key,
            )

        _seed_subject_indexes_bounded(
            problem_persistence=problem_persistence,
            bounded=bounded,
            inline_subject_refs=inline_subject_refs,
        )
        migrated.append(problem.problem_id)

    return ProblemOccurrenceMigrationPage(
        migrated_problem_ids=tuple(migrated),
        next_cursor=page.next_cursor,
        has_more=page.next_cursor is not None,
    )


_SUBJECT_INDEX_BATCH = 50


def _seed_subject_indexes_bounded(
    *,
    problem_persistence: ProblemPersistence,
    bounded: Problem,
    inline_subject_refs: tuple,
) -> None:
    """Seed subject ownership indexes in bounded batches (not one 100k-arg update)."""
    if not inline_subject_refs:
        return
    if isinstance(problem_persistence, DocumentStoreProblemPersistence):
        partition_key = _document_partition(bounded.tenant_id)
        for offset in range(0, len(inline_subject_refs), _SUBJECT_INDEX_BATCH):
            batch = inline_subject_refs[offset : offset + _SUBJECT_INDEX_BATCH]
            for subject_ref in batch:
                problem_persistence._ensure_subject_index(
                    record=bounded,
                    subject_ref=subject_ref,
                    partition_key=partition_key,
                )
        return
    for offset in range(0, len(inline_subject_refs), _SUBJECT_INDEX_BATCH):
        batch = inline_subject_refs[offset : offset + _SUBJECT_INDEX_BATCH]
        latest = problem_persistence.get(
            tenant_id=bounded.tenant_id,
            problem_id=bounded.problem_id,
        )
        if latest is None:
            raise RuntimeError("bounded Problem missing during migration index seed")
        problem_persistence.update(
            latest,
            expected_version=latest.record_version,
            indexed_subject_refs=batch,
        )


def verify_legacy_occurrences_migrated(
    *,
    tenant_id: str,
    problem_id: ProblemId,
    occurrence_persistence: ProblemOccurrencePersistence,
    document_store: ConditionalDocumentStore,
) -> bool:
    from intergrax.runtime.diagnostics.problem_occurrence_aggregate_reconciliation import (
        scan_occurrence_aggregate,
    )

    partition_key = _document_partition(tenant_id)
    record = document_store.get(partition_key, _record_row_key(problem_id))
    if record is None:
        return False
    problem = decode_problem_record(dict(record.data))
    scan = scan_occurrence_aggregate(
        occurrence_persistence,
        tenant_id=tenant_id,
        problem_id=problem_id,
    )
    if scan.occurrence_count == 0:
        return problem.occurrence_count == 0
    return scan.occurrence_count == problem.occurrence_count
