# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PartitionAtomicDocumentStore-backed execution lineage persistence (DG-001 R1)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionPage,
    ExecutionLineageAdmissionRecord,
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineageConfigurationError,
    ExecutionLineagePersistence,
    ExecutionLineageSealRecord,
    ExecutionLineageSegmentRecord,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentDataSort,
    DocumentQueryCursorCodec,
    DocumentRecord,
)
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicDocumentStore,
)
from intergrax.runtime.diagnostics.document_store_problem_occurrence_persistence import (
    DocumentStoreQueryCursorProvider,
)
from intergrax.runtime.execution.lineage.persistence import (
    _ExecutionLineageStoreLogic,
    _PartitionRow,
)


class _DocumentStorePartitionRowStore:
    def __init__(
        self,
        document_store: PartitionAtomicDocumentStore,
        *,
        query_cursor_codec: DocumentQueryCursorCodec,
    ) -> None:
        self._document_store = document_store
        self._query_cursor_codec = query_cursor_codec

    def get_row(self, partition_key: str, row_key: str) -> _PartitionRow | None:
        record = self._document_store.get(partition_key, row_key)
        if record is None:
            return None
        return _PartitionRow(partition_key, row_key, dict(record.data))

    def put_if_absent(self, row: _PartitionRow) -> bool:
        document = DocumentRecord(
            partition_key=row.partition_key,
            row_key=row.row_key,
            data=row.data,
        )
        return self._document_store.put_if_absent(document)

    def replace_if_match(self, expected: _PartitionRow, replacement: _PartitionRow) -> bool:
        return self._document_store.replace_if_match(
            expected=DocumentRecord(
                partition_key=expected.partition_key,
                row_key=expected.row_key,
                data=expected.data,
            ),
            replacement=DocumentRecord(
                partition_key=replacement.partition_key,
                row_key=replacement.row_key,
                data=replacement.data,
            ),
        )

    def list_rows(
        self,
        partition_key: str,
        *,
        row_key_prefix: str,
        limit: int,
        cursor: str | None,
        sort_path: str,
    ) -> tuple[tuple[_PartitionRow, ...], str | None]:
        page = self._document_store.query(
            partition_key,
            limit=limit,
            row_key_prefix=row_key_prefix,
            cursor=cursor,
            sort=(DocumentDataSort(path=sort_path, direction="asc"),),
        )
        rows = tuple(
            _PartitionRow(partition_key, document.row_key, dict(document.data))
            for document in page.documents
        )
        return rows, page.next_cursor


class DocumentStoreExecutionLineagePersistence(ExecutionLineagePersistence):
    """Durable execution lineage persistence over PartitionAtomicDocumentStore."""

    def __init__(
        self,
        document_store: PartitionAtomicDocumentStore,
        *,
        document_query_cursor_codec: DocumentQueryCursorCodec | None = None,
    ) -> None:
        if not isinstance(document_store, PartitionAtomicDocumentStore):
            raise ExecutionLineageConfigurationError(
                "execution lineage persistence requires PartitionAtomicDocumentStore",
            )
        if not isinstance(document_store, ConditionalDocumentStore):
            raise ExecutionLineageConfigurationError(
                "execution lineage persistence requires ConditionalDocumentStore",
            )
        cursor_codec = document_query_cursor_codec
        if cursor_codec is None and isinstance(document_store, DocumentStoreQueryCursorProvider):
            cursor_codec = document_store.query_cursor_codec
        if cursor_codec is None:
            raise ExecutionLineageConfigurationError(
                "execution lineage persistence requires document query cursor codec",
            )
        self._logic = _ExecutionLineageStoreLogic(
            _DocumentStorePartitionRowStore(document_store, query_cursor_codec=cursor_codec),
        )

    @property
    def is_durable(self) -> bool:
        return True

    def open_attempt(self, scope: ExecutionLineageAttemptScope) -> ExecutionLineageAttemptState:
        return self._logic.open_attempt(scope)

    def open_segment(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
        predecessor_root_execution_id: ExecutionId | None = None,
    ) -> ExecutionLineageSegmentRecord:
        return self._logic.open_segment(scope, root_execution_id, predecessor_root_execution_id)

    def admit_root(
        self,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        *,
        graph_node_id: str | None = None,
    ) -> ExecutionLineageAdmissionRecord:
        return self._logic.admit_root(
            scope,
            segment_root_execution_id,
            execution_id,
            graph_node_id=graph_node_id,
        )

    def admit_child(
        self,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        *,
        graph_node_id: str | None = None,
    ) -> ExecutionLineageAdmissionRecord:
        return self._logic.admit_child(
            scope,
            segment_root_execution_id,
            execution_id,
            parent_execution_id,
            graph_node_id=graph_node_id,
        )

    def close_segment_for_resume(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
    ) -> ExecutionLineageSegmentRecord:
        return self._logic.close_segment_for_resume(scope, root_execution_id)

    def mark_degraded(
        self,
        scope: ExecutionLineageAttemptScope,
        reason_code: str,
    ) -> ExecutionLineageAttemptState:
        return self._logic.mark_degraded(scope, reason_code)

    def seal_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        closure_kind: ExecutionLineageAttemptClosureKind,
    ) -> ExecutionLineageSealRecord:
        return self._logic.seal_attempt(scope, closure_kind)

    def list_admissions_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAdmissionPage:
        return self._logic.list_admissions_for_attempt(scope, limit, cursor=cursor)

    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        return self._logic.read_attempt_lineage_state(scope)

    def read_seal(self, scope: ExecutionLineageAttemptScope) -> ExecutionLineageSealRecord | None:
        return self._logic.read_seal(scope)
