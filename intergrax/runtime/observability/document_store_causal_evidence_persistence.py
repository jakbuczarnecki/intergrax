# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ConditionalDocumentStore-backed causal evidence persistence (DIAG-1D)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePersistence,
    CausalEvidencePersistenceConflictError,
)
from intergrax.runtime.observability.causal_evidence_record_codec import (
    decode_causal_evidence_record,
    encode_causal_evidence_record,
)

_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.causal_evidence.v1"
_RECORD_ROW_PREFIX = "record:"
_EXEC_ROW_PREFIX = "exec:"
_TRANSPORT_ROW_PREFIX = "transport:"
_QUERY_PAGE_LIMIT = 5000


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _record_row_key(evidence_id: str) -> str:
    return f"{_RECORD_ROW_PREFIX}{evidence_id}"


def _execution_row_key(*, task_id: TaskId, run_id: RunId, evidence_id: str) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:{evidence_id}"


def _transport_row_key(*, provider: str, transport_task_id: str, evidence_id: str) -> str:
    return f"{_TRANSPORT_ROW_PREFIX}{provider}:{transport_task_id}:{evidence_id}"


def _execution_row_prefix(*, task_id: TaskId, run_id: RunId) -> str:
    return f"{_EXEC_ROW_PREFIX}{task_id}:{run_id}:"


def _transport_row_prefix(*, provider: str, transport_task_id: str) -> str:
    return f"{_TRANSPORT_ROW_PREFIX}{provider}:{transport_task_id}:"


def _sort_key(evidence: PlatformCausalEvidence) -> tuple[str, str]:
    return (evidence.recorded_at.isoformat(), str(evidence.evidence_id))


class DocumentStoreCausalEvidencePersistence(CausalEvidencePersistence):
    """ConditionalDocumentStore-backed append-only causal evidence store."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "causal evidence persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    def append(self, evidence: PlatformCausalEvidence) -> PlatformCausalEvidence:
        partition_key = _document_partition(evidence.tenant_id)
        record_row_key = _record_row_key(str(evidence.evidence_id))
        encoded = encode_causal_evidence_record(evidence)

        existing_record = self._document_store.get(partition_key, record_row_key)
        if existing_record is not None:
            return self._resolve_existing_record(existing_record, evidence)

        exec_document = DocumentRecord(
            partition_key=partition_key,
            row_key=_execution_row_key(
                task_id=evidence.target.task_id,
                run_id=evidence.target.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=encoded,
        )
        transport_document = DocumentRecord(
            partition_key=partition_key,
            row_key=_transport_row_key(
                provider=evidence.source.provider,
                transport_task_id=evidence.source.task_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=encoded,
        )
        canonical_document = DocumentRecord(
            partition_key=partition_key,
            row_key=record_row_key,
            data=encoded,
        )

        if not self._document_store.put_if_absent(exec_document):
            self._verify_index_document(exec_document, evidence)
        if not self._document_store.put_if_absent(transport_document):
            self._verify_index_document(transport_document, evidence)

        if self._document_store.put_if_absent(canonical_document):
            return evidence

        raced = self._document_store.get(partition_key, record_row_key)
        if raced is None:
            raise RuntimeError("causal evidence persistence append failed")
        return self._resolve_existing_record(raced, evidence)

    def list_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> tuple[PlatformCausalEvidence, ...]:
        partition_key = _document_partition(tenant_id)
        prefix = _execution_row_prefix(task_id=task_id, run_id=run_id)
        return self._list_indexed(partition_key=partition_key, row_key_prefix=prefix)

    def list_for_transport_task(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
    ) -> tuple[PlatformCausalEvidence, ...]:
        partition_key = _document_partition(tenant_id)
        prefix = _transport_row_prefix(
            provider=provider,
            transport_task_id=transport_task_id,
        )
        return self._list_indexed(partition_key=partition_key, row_key_prefix=prefix)

    def _list_indexed(
        self,
        *,
        partition_key: str,
        row_key_prefix: str,
    ) -> tuple[PlatformCausalEvidence, ...]:
        documents: list[DocumentRecord] = []
        cursor: str | None = None
        while True:
            page = self._document_store.query(
                partition_key,
                limit=_QUERY_PAGE_LIMIT,
                row_key_prefix=row_key_prefix,
                cursor=cursor,
            )
            documents.extend(page.documents)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        decoded: list[PlatformCausalEvidence] = []
        for document in documents:
            decoded.append(self._document_to_evidence(document))
        decoded.sort(key=_sort_key)
        return tuple(decoded)

    def _document_to_evidence(self, document: DocumentRecord) -> PlatformCausalEvidence:
        return decode_causal_evidence_record(dict(document.data))

    def _resolve_existing_record(
        self,
        record: DocumentRecord,
        incoming: PlatformCausalEvidence,
    ) -> PlatformCausalEvidence:
        stored = self._document_to_evidence(record)
        if stored != incoming:
            raise CausalEvidencePersistenceConflictError(
                "conflicting causal evidence for evidence_id",
            )
        return stored

    def _verify_index_document(
        self,
        document: DocumentRecord,
        incoming: PlatformCausalEvidence,
    ) -> None:
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise RuntimeError("causal evidence index verification failed")
        self._resolve_existing_record(existing, incoming)


def wire_causal_evidence_persistence(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> CausalEvidencePersistence:
    """Platform composition boundary: storage capability → causal evidence persistence."""
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_causal_evidence_persistence accepts kv_store or "
            "document_store, not both",
        )
    if kv_store is not None:
        raise ValueError(
            "wire_causal_evidence_persistence does not support kv_store: "
            "DistributedKVStore lacks prefix-query primitives required for "
            "indexed causal evidence reads",
        )
    if document_store is not None:
        return DocumentStoreCausalEvidencePersistence(document_store)
    raise ValueError(
        "wire_causal_evidence_persistence requires document_store",
    )
