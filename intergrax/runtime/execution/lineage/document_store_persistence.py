# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PartitionAtomicDocumentStore-backed execution lineage persistence (DG-001 R1)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import AttemptId, ExecutionId
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionPage,
    ExecutionLineageAdmissionRecord,
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageAttemptDiscoveryPage,
    ExecutionLineageAttemptDiscoveryRecord,
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineageConfigurationError,
    ExecutionLineageDiscoveryRunState,
    ExecutionLineagePersistence,
    ExecutionLineageRunScope,
    ExecutionLineageSealRecord,
    ExecutionLineageSegmentPage,
    ExecutionLineageSegmentRecord,
    ExecutionLineageUnavailableError,
)
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentDataSort,
    DocumentQueryCursorCodec,
    DocumentRecord,
)
from intergrax.integrations.contracts.document_store_query_cursor_provider import (
    DocumentStoreQueryCursorProvider,
)
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicBatch,
    PartitionAtomicBatchResult,
    PartitionAtomicDocumentStore,
    PartitionPutIfAbsentOnCreated,
    PartitionReplaceIfMatchOnCreated,
)
from intergrax.runtime.execution.lineage.persistence import (
    _ExecutionLineageStoreLogic,
    _PartitionAtomicRowBatch,
    _PartitionAtomicRowBatchResult,
    _PartitionPutIfAbsentOnCreated,
    _PartitionReplaceIfMatchOnCreated,
    _PartitionRow,
)


def _row_to_document(row: _PartitionRow) -> DocumentRecord:
    return DocumentRecord(
        partition_key=row.partition_key,
        row_key=row.row_key,
        data=row.data,
    )


def _document_to_row(document: DocumentRecord) -> _PartitionRow:
    return _PartitionRow(document.partition_key, document.row_key, dict(document.data))


def _row_batch_to_document_batch(
    batch: _PartitionAtomicRowBatch,
) -> PartitionAtomicBatch:
    on_created_ops: list[
        PartitionPutIfAbsentOnCreated | PartitionReplaceIfMatchOnCreated
    ] = []
    for op in batch.on_created_ops:
        if isinstance(op, _PartitionPutIfAbsentOnCreated):
            on_created_ops.append(
                PartitionPutIfAbsentOnCreated(document=_row_to_document(op.row)),
            )
        elif isinstance(op, _PartitionReplaceIfMatchOnCreated):
            on_created_ops.append(
                PartitionReplaceIfMatchOnCreated(
                    expected=_row_to_document(op.expected),
                    replacement=_row_to_document(op.replacement),
                ),
            )
        else:
            raise TypeError("partition_atomic_row_batch_on_created_op_invalid")
    return PartitionAtomicBatch(
        partition_key=batch.partition_key,
        primary_put_if_absent=_row_to_document(batch.primary_put_if_absent),
        on_created_ops=tuple(on_created_ops),
    )


class _DocumentStorePartitionAtomicRowStore:
    def __init__(
        self,
        document_store: PartitionAtomicDocumentStore,
        *,
        query_cursor_codec: DocumentQueryCursorCodec,
    ) -> None:
        self._document_store = document_store
        self._query_cursor_codec = query_cursor_codec

    def get_row(self, partition_key: str, row_key: str) -> _PartitionRow | None:
        try:
            record = self._document_store.get(partition_key, row_key)
        except (OSError, RuntimeError) as exc:
            raise ExecutionLineageUnavailableError(str(exc)) from exc
        if record is None:
            return None
        return _document_to_row(record)

    def put_if_absent(self, row: _PartitionRow) -> bool:
        return self._document_store.put_if_absent(_row_to_document(row))

    def replace_if_match(
        self, expected: _PartitionRow, replacement: _PartitionRow
    ) -> bool:
        return self._document_store.replace_if_match(
            expected=_row_to_document(expected),
            replacement=_row_to_document(replacement),
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
        try:
            page = self._document_store.query(
                partition_key,
                limit=limit,
                row_key_prefix=row_key_prefix,
                cursor=cursor,
                sort=(DocumentDataSort(path=sort_path, direction="asc"),),
            )
        except (OSError, RuntimeError) as exc:
            raise ExecutionLineageUnavailableError(str(exc)) from exc
        rows = tuple(_document_to_row(document) for document in page.documents)
        return rows, page.next_cursor

    def execute_partition_atomic_batch(
        self,
        batch: _PartitionAtomicRowBatch,
    ) -> _PartitionAtomicRowBatchResult:
        result: PartitionAtomicBatchResult = (
            self._document_store.execute_partition_atomic_batch(
                _row_batch_to_document_batch(batch),
            )
        )
        return _PartitionAtomicRowBatchResult(primary_created=result.primary_created)


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
        if cursor_codec is None and isinstance(
            document_store, DocumentStoreQueryCursorProvider
        ):
            cursor_codec = document_store.query_cursor_codec
        if cursor_codec is None:
            raise ExecutionLineageConfigurationError(
                "execution lineage persistence requires document query cursor codec",
            )
        self._logic = _ExecutionLineageStoreLogic(
            _DocumentStorePartitionAtomicRowStore(
                document_store,
                query_cursor_codec=cursor_codec,
            ),
        )

    @property
    def is_durable(self) -> bool:
        return True

    def open_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        *,
        discovery_contract_version: int | None = None,
    ) -> ExecutionLineageAttemptState:
        return self._logic.open_attempt(
            scope,
            discovery_contract_version=discovery_contract_version,
        )

    def register_attempt_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord:
        return self._logic.register_attempt_for_run(run_scope, attempt_id)

    def open_segment(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
        predecessor_root_execution_id: ExecutionId | None = None,
    ) -> ExecutionLineageSegmentRecord:
        return self._logic.open_segment(
            scope, root_execution_id, predecessor_root_execution_id
        )

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

    def list_segments_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageSegmentPage:
        return self._logic.list_segments_for_attempt(scope, limit, cursor=cursor)

    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        return self._logic.read_attempt_lineage_state(scope)

    def read_seal(
        self, scope: ExecutionLineageAttemptScope
    ) -> ExecutionLineageSealRecord | None:
        return self._logic.read_seal(scope)

    def read_discovery_run_state(
        self,
        run_scope: ExecutionLineageRunScope,
    ) -> ExecutionLineageDiscoveryRunState | None:
        return self._logic.read_discovery_run_state(run_scope)

    def list_attempts_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAttemptDiscoveryPage:
        return self._logic.list_attempts_for_run(run_scope, limit, cursor=cursor)

    def read_attempt_discovery_record(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord | None:
        return self._logic.read_attempt_discovery_record(run_scope, attempt_id)
