# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded v1→v2 execution-index projection rebuild (DIAG-FUNCTIONAL-READ-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.runtime.diagnostics.functional_evidence import PlatformFunctionalEvidence
from intergrax.runtime.diagnostics.functional_evidence_execution_index import (
    decode_execution_index_v1,
    encode_execution_index_v2,
    execution_index_v1_row_key_prefix,
    execution_index_v2_row_key_from_evidence,
    execution_index_v2_row_key_prefix,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.functional_evidence_record_codec import (
    decode_functional_evidence_record,
)

_QUERY_PAGE_LIMIT = 5000
_RECORD_ROW_PREFIX = "record:"


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceIndexRebuildResult:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    v2_entries_written: int
    v1_entries_scanned: int


class FunctionalEvidenceIndexRebuilder:
    """Idempotent, tenant-scoped rebuild of order-aware execution index projections."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        query_page_limit: int = _QUERY_PAGE_LIMIT,
    ) -> None:
        self._document_store = document_store
        if type(query_page_limit) is not int or isinstance(query_page_limit, bool):
            raise ValueError("functional_evidence_index_rebuild_page_limit_invalid")
        if query_page_limit < 1:
            raise ValueError("functional_evidence_index_rebuild_page_limit_invalid")
        self._query_page_limit = query_page_limit

    def ensure_v2_projection(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> FunctionalEvidenceIndexRebuildResult | None:
        v2_prefix = execution_index_v2_row_key_prefix(task_id=task_id, run_id=run_id)
        if self._document_store.query(
            partition_key,
            limit=1,
            row_key_prefix=v2_prefix,
        ).documents:
            return None

        v1_prefix = execution_index_v1_row_key_prefix(task_id=task_id, run_id=run_id)
        v1_probe = self._document_store.query(
            partition_key,
            limit=1,
            row_key_prefix=v1_prefix,
        )
        if not v1_probe.documents:
            return None

        return self.rebuild_execution_index(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            partition_key=partition_key,
        )

    def rebuild_execution_index(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> FunctionalEvidenceIndexRebuildResult:
        v1_prefix = execution_index_v1_row_key_prefix(task_id=task_id, run_id=run_id)
        scanned = 0
        written = 0
        continuation: str | None = None
        while True:
            page = self._document_store.query(
                partition_key,
                limit=self._query_page_limit,
                row_key_prefix=v1_prefix,
                cursor=continuation,
            )
            for index_document in page.documents:
                scanned += 1
                indexed = decode_execution_index_v1(dict(index_document.data))
                evidence = self._load_canonical_evidence(
                    partition_key=partition_key,
                    evidence_id=indexed.evidence_id,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    run_id=run_id,
                )
                v2_document = DocumentRecord(
                    partition_key=partition_key,
                    row_key=execution_index_v2_row_key_from_evidence(evidence),
                    data=encode_execution_index_v2(evidence),
                )
                if self._document_store.put_if_absent(v2_document):
                    written += 1
                    continue
                self._verify_v2_index_document(v2_document, evidence)
            if page.next_cursor is None:
                break
            continuation = page.next_cursor
        return FunctionalEvidenceIndexRebuildResult(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            v2_entries_written=written,
            v1_entries_scanned=scanned,
        )

    def _load_canonical_evidence(
        self,
        *,
        partition_key: str,
        evidence_id: str,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> PlatformFunctionalEvidence:
        record = self._document_store.get(partition_key, f"{_RECORD_ROW_PREFIX}{evidence_id}")
        if record is None:
            raise FunctionalEvidencePersistenceIntegrityError(
                "canonical functional evidence record missing for index",
            )
        try:
            evidence = decode_functional_evidence_record(dict(record.data))
        except ValueError as exc:
            raise FunctionalEvidencePersistenceIntegrityError(
                "invalid canonical functional evidence record",
            ) from exc
        if str(evidence.evidence_id) != evidence_id:
            raise FunctionalEvidencePersistenceIntegrityError(
                "canonical functional evidence id does not match index reference",
            )
        if (
            evidence.scope.tenant_id != tenant_id
            or evidence.scope.task_id != task_id
            or evidence.scope.run_id != run_id
        ):
            raise FunctionalEvidencePersistenceIntegrityError(
                "canonical functional evidence does not match execution index scope",
            )
        return evidence

    def _verify_v2_index_document(
        self,
        document: DocumentRecord,
        evidence: PlatformFunctionalEvidence,
    ) -> None:
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index verification failed",
            )
        if dict(existing.data) != dict(document.data):
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index conflicts with expected projection",
            )
        if str(evidence.evidence_id) not in document.row_key:
            raise FunctionalEvidencePersistenceIntegrityError(
                "functional evidence index conflicts with expected evidence_id",
            )


__all__ = [
    "FunctionalEvidenceIndexRebuildResult",
    "FunctionalEvidenceIndexRebuilder",
]
