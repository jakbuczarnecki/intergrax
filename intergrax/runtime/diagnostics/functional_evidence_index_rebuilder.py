# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded v1→v2 execution-index projection rebuild (DIAG-FUNCTIONAL-READ-R1 / R1-R1)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
)
from intergrax.runtime.diagnostics.functional_evidence import PlatformFunctionalEvidence
from intergrax.runtime.diagnostics.functional_evidence_execution_index import (
    decode_execution_index_v1,
    decode_execution_index_v2,
    encode_execution_index_v2,
    execution_index_v1_row_key,
    execution_index_v1_row_key_prefix,
    execution_index_v2_row_key_from_evidence,
    execution_index_v2_row_key_prefix,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.functional_evidence_projection_state import (
    FunctionalEvidenceProjectionStateStore,
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
    reconciliation_passes: int


@dataclass(frozen=True, slots=True)
class FunctionalEvidenceIndexReconcilePassResult:
    v2_entries_written: int
    v1_entries_scanned: int


class FunctionalEvidenceIndexRebuilder:
    """Idempotent, tenant-scoped rebuild of order-aware execution index projections."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        query_page_limit: int = _QUERY_PAGE_LIMIT,
        interrupt_after_v2_writes: Callable[[int], bool] | None = None,
    ) -> None:
        self._document_store = document_store
        if type(query_page_limit) is not int or isinstance(query_page_limit, bool):
            raise ValueError("functional_evidence_index_rebuild_page_limit_invalid")
        if query_page_limit < 1:
            raise ValueError("functional_evidence_index_rebuild_page_limit_invalid")
        self._query_page_limit = query_page_limit
        self._projection_state = FunctionalEvidenceProjectionStateStore(document_store)
        self._interrupt_after_v2_writes = interrupt_after_v2_writes

    def ensure_v2_projection(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> FunctionalEvidenceIndexRebuildResult | None:
        manifest = self._projection_state.load(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
        )
        if manifest is not None and manifest.state == "complete":
            return None

        v1_prefix = execution_index_v1_row_key_prefix(task_id=task_id, run_id=run_id)
        v2_prefix = execution_index_v2_row_key_prefix(task_id=task_id, run_id=run_id)
        has_v1 = bool(
            self._document_store.query(
                partition_key,
                limit=1,
                row_key_prefix=v1_prefix,
            ).documents,
        )
        has_v2 = bool(
            self._document_store.query(
                partition_key,
                limit=1,
                row_key_prefix=v2_prefix,
            ).documents,
        )
        if not has_v1 and not has_v2:
            return None
        if has_v1:
            return self.rebuild_execution_index(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                partition_key=partition_key,
            )
        self._verify_v2_orphans(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            partition_key=partition_key,
        )
        return None

    def rebuild_execution_index(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> FunctionalEvidenceIndexRebuildResult:
        generation = self._projection_state.begin_rebuild(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
        )
        total_written = 0
        total_scanned = 0
        passes = 0
        while True:
            passes += 1
            pass_result = self._reconcile_v1_to_v2_pass(
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
                partition_key=partition_key,
            )
            total_written += pass_result.v2_entries_written
            total_scanned = pass_result.v1_entries_scanned
            if pass_result.v2_entries_written == 0:
                break
        self._verify_v2_orphans(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            partition_key=partition_key,
        )
        self._projection_state.mark_complete(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
            generation=generation,
            v1_rows_reconciled=total_scanned,
        )
        return FunctionalEvidenceIndexRebuildResult(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            v2_entries_written=total_written,
            v1_entries_scanned=total_scanned,
            reconciliation_passes=passes,
        )

    def _reconcile_v1_to_v2_pass(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> FunctionalEvidenceIndexReconcilePassResult:
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
                    if (
                        self._interrupt_after_v2_writes is not None
                        and self._interrupt_after_v2_writes(written)
                    ):
                        raise FunctionalEvidencePersistenceIntegrityError(
                            "functional evidence projection rebuild interrupted",
                        )
                    continue
                self._verify_v2_index_document(v2_document, evidence)
            if page.next_cursor is None:
                break
            continuation = page.next_cursor
        return FunctionalEvidenceIndexReconcilePassResult(
            v2_entries_written=written,
            v1_entries_scanned=scanned,
        )

    def _verify_v2_orphans(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> None:
        v2_prefix = execution_index_v2_row_key_prefix(task_id=task_id, run_id=run_id)
        continuation: str | None = None
        while True:
            page = self._document_store.query(
                partition_key,
                limit=self._query_page_limit,
                row_key_prefix=v2_prefix,
                cursor=continuation,
            )
            for index_document in page.documents:
                indexed = decode_execution_index_v2(dict(index_document.data))
                v1_row_key = execution_index_v1_row_key(
                    task_id=task_id,
                    run_id=run_id,
                    evidence_id=indexed.evidence_id,
                )
                v1_record = self._document_store.get(partition_key, v1_row_key)
                if v1_record is None:
                    raise FunctionalEvidencePersistenceIntegrityError(
                        "functional evidence execution index v2 row has no matching v1 projection",
                    )
                self._load_canonical_evidence(
                    partition_key=partition_key,
                    evidence_id=indexed.evidence_id,
                    tenant_id=tenant_id,
                    task_id=task_id,
                    run_id=run_id,
                )
            if page.next_cursor is None:
                break
            continuation = page.next_cursor

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
    "FunctionalEvidenceIndexReconcilePassResult",
]
