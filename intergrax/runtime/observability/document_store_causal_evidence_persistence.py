# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ConditionalDocumentStore-backed causal evidence persistence (DIAG-1D)."""

from __future__ import annotations

from collections.abc import Callable

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
    CausalEvidencePersistenceIntegrityError,
    causal_evidence_query_order_key,
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
_INDEX_SCHEMA = "intergrax.causal_evidence.index.v1"
_EVIDENCE_ID_FIELD = "evidence_id"
_LEGACY_RECORD_SCHEMA = "intergrax.causal_evidence.persistence.v1"


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


def _encode_index_ref(evidence_id: str) -> dict[str, str]:
    return {
        "schema_version": _INDEX_SCHEMA,
        _EVIDENCE_ID_FIELD: evidence_id,
    }


def _decode_index_ref(data: object) -> str:
    if not isinstance(data, dict):
        raise CausalEvidencePersistenceIntegrityError("invalid causal evidence index")
    schema_version = data.get("schema_version")
    if schema_version == _INDEX_SCHEMA:
        evidence_id = data.get(_EVIDENCE_ID_FIELD)
        if not isinstance(evidence_id, str) or not evidence_id:
            raise CausalEvidencePersistenceIntegrityError(
                "invalid causal evidence index reference",
            )
        return evidence_id
    if schema_version == _LEGACY_RECORD_SCHEMA:
        return str(decode_causal_evidence_record(data).evidence_id)
    raise CausalEvidencePersistenceIntegrityError(
        "unsupported causal evidence index schema",
    )


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
        canonical_document = DocumentRecord(
            partition_key=partition_key,
            row_key=record_row_key,
            data=encoded,
        )

        if self._document_store.put_if_absent(canonical_document):
            self._ensure_indexes(evidence=evidence, partition_key=partition_key)
            return evidence

        existing_record = self._document_store.get(partition_key, record_row_key)
        if existing_record is None:
            raise RuntimeError("causal evidence persistence append failed")
        return self._resolve_existing_record_and_repair_indexes(
            existing_record,
            evidence,
            partition_key=partition_key,
        )

    def list_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> tuple[PlatformCausalEvidence, ...]:
        partition_key = _document_partition(tenant_id)
        prefix = _execution_row_prefix(task_id=task_id, run_id=run_id)

        def _validate_scope(evidence: PlatformCausalEvidence) -> None:
            if (
                evidence.tenant_id != tenant_id
                or evidence.target.task_id != task_id
                or evidence.target.run_id != run_id
            ):
                raise CausalEvidencePersistenceIntegrityError(
                    "canonical causal evidence does not match execution index scope",
                )

        return self._list_indexed(
            partition_key=partition_key,
            row_key_prefix=prefix,
            validate_scope=_validate_scope,
        )

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

        def _validate_scope(evidence: PlatformCausalEvidence) -> None:
            if (
                evidence.tenant_id != tenant_id
                or evidence.source.provider != provider
                or evidence.source.task_id != transport_task_id
            ):
                raise CausalEvidencePersistenceIntegrityError(
                    "canonical causal evidence does not match transport index scope",
                )

        return self._list_indexed(
            partition_key=partition_key,
            row_key_prefix=prefix,
            validate_scope=_validate_scope,
        )

    def _list_indexed(
        self,
        *,
        partition_key: str,
        row_key_prefix: str,
        validate_scope: Callable[[PlatformCausalEvidence], None],
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
            evidence_id = _decode_index_ref(dict(document.data))
            record = self._document_store.get(
                partition_key,
                _record_row_key(evidence_id),
            )
            if record is None:
                raise CausalEvidencePersistenceIntegrityError(
                    "canonical causal evidence record missing for index",
                )
            evidence = decode_causal_evidence_record(dict(record.data))
            if str(evidence.evidence_id) != evidence_id:
                raise CausalEvidencePersistenceIntegrityError(
                    "canonical causal evidence id does not match index reference",
                )
            validate_scope(evidence)
            decoded.append(evidence)
        decoded.sort(key=causal_evidence_query_order_key)
        return tuple(decoded)

    def _execution_index_document(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition_key,
            row_key=_execution_row_key(
                task_id=evidence.target.task_id,
                run_id=evidence.target.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=_encode_index_ref(str(evidence.evidence_id)),
        )

    def _transport_index_document(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> DocumentRecord:
        return DocumentRecord(
            partition_key=partition_key,
            row_key=_transport_row_key(
                provider=evidence.source.provider,
                transport_task_id=evidence.source.task_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=_encode_index_ref(str(evidence.evidence_id)),
        )

    def _ensure_indexes(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> None:
        exec_document = self._execution_index_document(
            evidence=evidence,
            partition_key=partition_key,
        )
        transport_document = self._transport_index_document(
            evidence=evidence,
            partition_key=partition_key,
        )
        if not self._document_store.put_if_absent(exec_document):
            self._verify_index_document(exec_document, evidence)
        if not self._document_store.put_if_absent(transport_document):
            self._verify_index_document(transport_document, evidence)

    def _document_to_evidence(self, document: DocumentRecord) -> PlatformCausalEvidence:
        return decode_causal_evidence_record(dict(document.data))

    def _resolve_existing_record_and_repair_indexes(
        self,
        record: DocumentRecord,
        incoming: PlatformCausalEvidence,
        *,
        partition_key: str,
    ) -> PlatformCausalEvidence:
        stored = self._document_to_evidence(record)
        if stored != incoming:
            raise CausalEvidencePersistenceConflictError(
                "conflicting causal evidence for evidence_id",
            )
        self._ensure_indexes(evidence=stored, partition_key=partition_key)
        return stored

    def _verify_index_document(
        self,
        document: DocumentRecord,
        incoming: PlatformCausalEvidence,
    ) -> None:
        existing = self._document_store.get(document.partition_key, document.row_key)
        if existing is None:
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence index verification failed",
            )
        indexed_id = _decode_index_ref(dict(existing.data))
        if indexed_id != str(incoming.evidence_id):
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence index conflicts with expected evidence_id",
            )


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
