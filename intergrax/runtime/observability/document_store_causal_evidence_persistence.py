# © Artur Czarnecki. All rights reserved.

# Intergrax framework – proprietary and confidential.


"""ConditionalDocumentStore-backed causal evidence persistence (DIAG-1D)."""

from __future__ import annotations


import secrets

from collections.abc import Callable

from typing import Protocol, runtime_checkable


from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)

from intergrax.distributed.contracts.kv_store import DistributedKVStore

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentDataSort,
    DocumentQueryCursorCodec,
    DocumentRecord,
    DocumentStore,
)

from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence

from intergrax.runtime.observability.causal_evidence_index import (
    decode_causal_evidence_index_v1,
    decode_causal_evidence_index_v2,
    encode_causal_evidence_index_v1,
    encode_causal_evidence_index_v2,
    execution_index_v1_row_key,
    execution_index_v1_row_key_prefix,
    execution_index_v2_row_key_from_evidence,
    execution_index_v2_row_key_prefix,
    is_v2_index_row_key,
    transport_index_v1_row_key,
    transport_index_v1_row_key_prefix,
    transport_index_v2_row_key_from_evidence,
    transport_index_v2_row_key_prefix,
    v2_index_matches_row_key,
)

from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePage,
    CausalEvidencePersistence,
    CausalEvidencePersistenceConflictError,
    CausalEvidencePersistenceIntegrityError,
    causal_evidence_query_order_key,
    validate_causal_evidence_query_limit,
)

from intergrax.runtime.observability.causal_evidence_query_cursor import (
    CausalEvidenceQueryCursorCodec,
    CausalEvidenceQueryCursorError,
)

from intergrax.runtime.observability.causal_evidence_record_codec import (
    decode_causal_evidence_record,
    encode_causal_evidence_record,
)


_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.causal_evidence.v1"

_RECORD_ROW_PREFIX = "record:"

_QUERY_PAGE_LIMIT = 5000

_RECONCILE_PAGE_LIMIT = 100

_LEGACY_RECORD_SCHEMA = "intergrax.causal_evidence.persistence.v1"

_SCOPE_META_SCHEMA = "intergrax.causal_evidence.scope_meta.v1"

_V2_READY_VALUE = "ready"

_HIGH_WATER_FIELD = "row_key"

_V2_READY_EXEC_PREFIX = "meta:v2_ready:exec:"

_V2_READY_TRANSPORT_PREFIX = "meta:v2_ready:transport:"

_HIGH_WATER_EXEC_PREFIX = "meta:high_water:exec:"

_HIGH_WATER_TRANSPORT_PREFIX = "meta:high_water:transport:"

_RECONCILE_EXEC_PREFIX = "meta:v1_reconcile:exec:"

_RECONCILE_TRANSPORT_PREFIX = "meta:v1_reconcile:transport:"

_ROW_KEY_SORT_DESC = (DocumentDataSort(path="$row_key", direction="desc"),)


@runtime_checkable
class DocumentStoreQueryCursorProvider(Protocol):
    @property
    def query_cursor_codec(self) -> DocumentQueryCursorCodec:
        """Authenticated codec for document-store query continuation cursors."""


def _document_partition(tenant_id: str) -> str:

    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _record_row_key(evidence_id: str) -> str:

    return f"{_RECORD_ROW_PREFIX}{evidence_id}"


def _v2_ready_exec_row_key(*, task_id: TaskId, run_id: RunId) -> str:

    return f"{_V2_READY_EXEC_PREFIX}{task_id}:{run_id}"


def _v2_ready_transport_row_key(*, provider: str, transport_task_id: str) -> str:

    return f"{_V2_READY_TRANSPORT_PREFIX}{provider}:{transport_task_id}"


def _high_water_exec_row_key(*, task_id: TaskId, run_id: RunId) -> str:

    return f"{_HIGH_WATER_EXEC_PREFIX}{task_id}:{run_id}"


def _high_water_transport_row_key(*, provider: str, transport_task_id: str) -> str:

    return f"{_HIGH_WATER_TRANSPORT_PREFIX}{provider}:{transport_task_id}"


def _reconcile_exec_row_key(*, task_id: TaskId, run_id: RunId) -> str:

    return f"{_RECONCILE_EXEC_PREFIX}{task_id}:{run_id}"


def _reconcile_transport_row_key(*, provider: str, transport_task_id: str) -> str:

    return f"{_RECONCILE_TRANSPORT_PREFIX}{provider}:{transport_task_id}"


def _encode_scope_marker(*, row_key: str | None = None) -> dict[str, str]:

    payload = {"schema_version": _SCOPE_META_SCHEMA}

    if row_key is not None:
        payload[_HIGH_WATER_FIELD] = row_key

    else:
        payload["value"] = _V2_READY_VALUE

    return payload


def _decode_scope_row_key(data: object) -> str:

    if not isinstance(data, dict):
        raise CausalEvidencePersistenceIntegrityError(
            "invalid causal evidence scope marker"
        )

    if data.get("schema_version") != _SCOPE_META_SCHEMA:
        raise CausalEvidencePersistenceIntegrityError(
            "invalid causal evidence scope marker"
        )

    row_key = data.get(_HIGH_WATER_FIELD)

    if not isinstance(row_key, str) or not row_key:
        raise CausalEvidencePersistenceIntegrityError(
            "invalid causal evidence high-water marker"
        )

    return row_key


def _translate_document_store_cursor_error(
    exc: ValueError,
) -> CausalEvidencePersistenceIntegrityError:

    message = str(exc)

    if message in {
        "document_store_cursor_invalid",
        "document_store_cursor_query_mismatch",
        "document_store_cursor_authentication_failed",
    }:
        return CausalEvidencePersistenceIntegrityError(message)

    raise exc


class DocumentStoreCausalEvidencePersistence(CausalEvidencePersistence):
    """ConditionalDocumentStore-backed append-only causal evidence store."""

    def __init__(
        self,
        document_store: ConditionalDocumentStore,
        *,
        cursor_secret: bytes | None = None,
        document_query_cursor_codec: DocumentQueryCursorCodec | None = None,
    ) -> None:

        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "causal evidence persistence requires ConditionalDocumentStore",
            )

        self._document_store = document_store

        self._cursor_codec = CausalEvidenceQueryCursorCodec(
            secret=cursor_secret or secrets.token_bytes(32),
        )

        self._document_query_cursor_codec = self._resolve_document_query_cursor_codec(
            document_store,
            document_query_cursor_codec,
        )

    @staticmethod
    def _resolve_document_query_cursor_codec(
        document_store: ConditionalDocumentStore,
        document_query_cursor_codec: DocumentQueryCursorCodec | None,
    ) -> DocumentQueryCursorCodec:

        if document_query_cursor_codec is not None:
            return document_query_cursor_codec

        if isinstance(document_store, DocumentStoreQueryCursorProvider):
            return document_store.query_cursor_codec

        raise TypeError(
            "causal evidence persistence requires document store query cursor codec",
        )

    @property
    def query_cursor_codec(self) -> CausalEvidenceQueryCursorCodec:

        return self._cursor_codec

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

    def page_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        limit: int,
        cursor: str | None = None,
    ) -> CausalEvidencePage:

        validated_limit = validate_causal_evidence_query_limit(limit)

        validated_task_id = validate_task_id(task_id)

        validated_run_id = validate_run_id(run_id)

        partition_key = _document_partition(tenant_id)

        row_prefix = execution_index_v2_row_key_prefix(
            task_id=validated_task_id,
            run_id=validated_run_id,
        )

        self._ensure_execution_scope_ready_for_paging(
            tenant_id=tenant_id,
            task_id=validated_task_id,
            run_id=validated_run_id,
            partition_key=partition_key,
        )

        return self._page_v2_scope(
            partition_key=partition_key,
            row_prefix=row_prefix,
            limit=validated_limit,
            cursor=cursor,
            decode_cursor=lambda value: self._cursor_codec.decode_execution(
                value,
                tenant_id=tenant_id,
                task_id=validated_task_id,
                run_id=validated_run_id,
            ),
            encode_cursor=lambda **kwargs: self._cursor_codec.encode_execution(
                tenant_id=tenant_id,
                task_id=validated_task_id,
                run_id=validated_run_id,
                **kwargs,
            ),
            read_high_water=lambda: self._read_high_water(
                partition_key=partition_key,
                marker_row_key=_high_water_exec_row_key(
                    task_id=validated_task_id,
                    run_id=validated_run_id,
                ),
            ),
            validate_scope=lambda evidence: self._validate_execution_scope(
                evidence,
                tenant_id=tenant_id,
                task_id=validated_task_id,
                run_id=validated_run_id,
            ),
        )

    def page_for_transport_task(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> CausalEvidencePage:

        validated_limit = validate_causal_evidence_query_limit(limit)

        partition_key = _document_partition(tenant_id)

        row_prefix = transport_index_v2_row_key_prefix(
            provider=provider,
            transport_task_id=transport_task_id,
        )

        self._ensure_transport_scope_ready_for_paging(
            tenant_id=tenant_id,
            provider=provider,
            transport_task_id=transport_task_id,
            partition_key=partition_key,
        )

        return self._page_v2_scope(
            partition_key=partition_key,
            row_prefix=row_prefix,
            limit=validated_limit,
            cursor=cursor,
            decode_cursor=lambda value: self._cursor_codec.decode_transport(
                value,
                tenant_id=tenant_id,
                provider=provider,
                transport_task_id=transport_task_id,
            ),
            encode_cursor=lambda **kwargs: self._cursor_codec.encode_transport(
                tenant_id=tenant_id,
                provider=provider,
                transport_task_id=transport_task_id,
                **kwargs,
            ),
            read_high_water=lambda: self._read_high_water(
                partition_key=partition_key,
                marker_row_key=_high_water_transport_row_key(
                    provider=provider,
                    transport_task_id=transport_task_id,
                ),
            ),
            validate_scope=lambda evidence: self._validate_transport_scope(
                evidence,
                tenant_id=tenant_id,
                provider=provider,
                transport_task_id=transport_task_id,
            ),
        )

    def list_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> tuple[PlatformCausalEvidence, ...]:

        partition_key = _document_partition(tenant_id)

        prefix = execution_index_v1_row_key_prefix(task_id=task_id, run_id=run_id)

        def _validate_scope(evidence: PlatformCausalEvidence) -> None:

            self._validate_execution_scope(
                evidence,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
            )

        return self._list_legacy_aware(
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

        prefix = transport_index_v1_row_key_prefix(
            provider=provider,
            transport_task_id=transport_task_id,
        )

        def _validate_scope(evidence: PlatformCausalEvidence) -> None:

            self._validate_transport_scope(
                evidence,
                tenant_id=tenant_id,
                provider=provider,
                transport_task_id=transport_task_id,
            )

        return self._list_legacy_aware(
            partition_key=partition_key,
            row_key_prefix=prefix,
            validate_scope=_validate_scope,
        )

    def _page_v2_scope(
        self,
        *,
        partition_key: str,
        row_prefix: str,
        limit: int,
        cursor: str | None,
        decode_cursor,
        encode_cursor,
        read_high_water: Callable[[], str | None],
        validate_scope: Callable[[PlatformCausalEvidence], None],
    ) -> CausalEvidencePage:

        store_cursor: str | None = None

        high_water: str | None

        if cursor is None:
            high_water = self._derive_high_water(
                partition_key=partition_key,
                row_prefix=row_prefix,
                read_high_water=read_high_water,
            )

            if high_water is None:
                return CausalEvidencePage(items=(), next_cursor=None)

        else:
            try:
                payload = decode_cursor(cursor)

            except CausalEvidenceQueryCursorError as exc:
                raise CausalEvidencePersistenceIntegrityError(str(exc)) from exc

            high_water = payload.high_water

            store_cursor = payload.store_cursor

        try:
            page = self._document_store.query(
                partition_key,
                limit=limit,
                row_key_prefix=row_prefix,
                cursor=store_cursor,
                row_key_upper_bound=high_water,
            )

        except ValueError as exc:
            raise _translate_document_store_cursor_error(exc) from exc

        items: list[PlatformCausalEvidence] = []

        last_row_key: str | None = None

        last_recorded_at = None

        last_evidence_id: str | None = None

        for index_document in page.documents:
            if not is_v2_index_row_key(index_document.row_key, row_prefix=row_prefix):
                raise CausalEvidencePersistenceIntegrityError(
                    "non-v2 causal evidence index row in bounded page",
                )

            last_row_key = index_document.row_key

            evidence = self._resolve_v2_index_document(
                index_document,
                partition_key=partition_key,
                row_prefix=row_prefix,
                validate_scope=validate_scope,
            )

            items.append(evidence)

            last_recorded_at = evidence.recorded_at

            last_evidence_id = str(evidence.evidence_id)

        next_cursor: str | None = None

        if page.next_cursor is not None and last_row_key is not None:
            next_store_cursor = page.next_cursor

            if last_recorded_at is not None and last_evidence_id is not None:
                next_cursor = encode_cursor(
                    high_water=high_water,
                    last_recorded_at=last_recorded_at,
                    last_evidence_id=last_evidence_id,
                    store_cursor=next_store_cursor,
                )

        return CausalEvidencePage(items=tuple(items), next_cursor=next_cursor)

    def _derive_high_water(
        self,
        *,
        partition_key: str,
        row_prefix: str,
        read_high_water: Callable[[], str | None],
    ) -> str | None:

        marker = read_high_water()

        if marker is not None:
            return marker

        try:
            page = self._document_store.query(
                partition_key,
                limit=1,
                row_key_prefix=row_prefix,
                sort=_ROW_KEY_SORT_DESC,
            )

        except ValueError as exc:
            raise _translate_document_store_cursor_error(exc) from exc

        if not page.documents:
            return None

        return page.documents[0].row_key

    def _ensure_execution_scope_ready_for_paging(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> None:

        if self._is_execution_v2_ready(
            partition_key=partition_key,
            task_id=task_id,
            run_id=run_id,
        ):
            return

        completed = self._reconcile_v1_execution_page(
            tenant_id=tenant_id,
            task_id=task_id,
            run_id=run_id,
            partition_key=partition_key,
        )

        if not completed:
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence execution index reconciliation incomplete",
            )

    def _ensure_transport_scope_ready_for_paging(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        partition_key: str,
    ) -> None:

        if self._is_transport_v2_ready(
            partition_key=partition_key,
            provider=provider,
            transport_task_id=transport_task_id,
        ):
            return

        completed = self._reconcile_v1_transport_page(
            tenant_id=tenant_id,
            provider=provider,
            transport_task_id=transport_task_id,
            partition_key=partition_key,
        )

        if not completed:
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence transport index reconciliation incomplete",
            )

    def _reconcile_v1_execution_page(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        partition_key: str,
    ) -> bool:

        row_prefix = execution_index_v1_row_key_prefix(task_id=task_id, run_id=run_id)

        reconcile_key = _reconcile_exec_row_key(task_id=task_id, run_id=run_id)

        store_cursor = self._read_reconcile_cursor(partition_key, reconcile_key)

        try:
            page = self._document_store.query(
                partition_key,
                limit=_RECONCILE_PAGE_LIMIT,
                row_key_prefix=row_prefix,
                cursor=store_cursor,
            )

        except ValueError as exc:
            raise _translate_document_store_cursor_error(exc) from exc

        for index_document in page.documents:
            if is_v2_index_row_key(index_document.row_key, row_prefix=row_prefix):
                continue

            evidence_id = self._decode_index_ref(dict(index_document.data))

            evidence = self._load_canonical_evidence(
                partition_key=partition_key,
                evidence_id=evidence_id,
            )

            self._validate_execution_scope(
                evidence,
                tenant_id=tenant_id,
                task_id=task_id,
                run_id=run_id,
            )

            self._put_v2_execution_index(evidence=evidence, partition_key=partition_key)

        if page.next_cursor is None:
            self._mark_execution_v2_ready(
                partition_key=partition_key,
                task_id=task_id,
                run_id=run_id,
            )

            self._document_store.delete(partition_key, reconcile_key)

            return True

        next_store_cursor = page.next_cursor

        self._document_store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=reconcile_key,
                data=_encode_scope_marker(row_key=next_store_cursor),
            ),
        )

        return False

    def _reconcile_v1_transport_page(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        partition_key: str,
    ) -> bool:

        row_prefix = transport_index_v1_row_key_prefix(
            provider=provider,
            transport_task_id=transport_task_id,
        )

        reconcile_key = _reconcile_transport_row_key(
            provider=provider,
            transport_task_id=transport_task_id,
        )

        store_cursor = self._read_reconcile_cursor(partition_key, reconcile_key)

        try:
            page = self._document_store.query(
                partition_key,
                limit=_RECONCILE_PAGE_LIMIT,
                row_key_prefix=row_prefix,
                cursor=store_cursor,
            )

        except ValueError as exc:
            raise _translate_document_store_cursor_error(exc) from exc

        for index_document in page.documents:
            if is_v2_index_row_key(index_document.row_key, row_prefix=row_prefix):
                continue

            evidence_id = self._decode_index_ref(dict(index_document.data))

            evidence = self._load_canonical_evidence(
                partition_key=partition_key,
                evidence_id=evidence_id,
            )

            self._validate_transport_scope(
                evidence,
                tenant_id=tenant_id,
                provider=provider,
                transport_task_id=transport_task_id,
            )

            self._put_v2_transport_index(evidence=evidence, partition_key=partition_key)

        if page.next_cursor is None:
            self._mark_transport_v2_ready(
                partition_key=partition_key,
                provider=provider,
                transport_task_id=transport_task_id,
            )

            self._document_store.delete(partition_key, reconcile_key)

            return True

        self._document_store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=reconcile_key,
                data=_encode_scope_marker(row_key=page.next_cursor),
            ),
        )

        return False

    def _list_legacy_aware(
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

        decoded_by_id: dict[str, PlatformCausalEvidence] = {}

        for document in documents:
            if document.row_key.startswith("meta:"):
                continue

            evidence_id = self._decode_index_ref(dict(document.data))

            if evidence_id in decoded_by_id:
                continue

            evidence = self._load_canonical_evidence(
                partition_key=partition_key,
                evidence_id=evidence_id,
            )

            if str(evidence.evidence_id) != evidence_id:
                raise CausalEvidencePersistenceIntegrityError(
                    "canonical causal evidence id does not match index reference",
                )

            validate_scope(evidence)

            decoded_by_id[evidence_id] = evidence

        decoded = list(decoded_by_id.values())

        decoded.sort(key=causal_evidence_query_order_key)

        return tuple(decoded)

    def _resolve_v2_index_document(
        self,
        index_document: DocumentRecord,
        *,
        partition_key: str,
        row_prefix: str,
        validate_scope: Callable[[PlatformCausalEvidence], None],
    ) -> PlatformCausalEvidence:

        try:
            indexed = decode_causal_evidence_index_v2(dict(index_document.data))

        except ValueError as exc:
            raise CausalEvidencePersistenceIntegrityError(
                "invalid causal evidence v2 index",
            ) from exc

        if not v2_index_matches_row_key(
            indexed,
            row_key=index_document.row_key,
            row_prefix=row_prefix,
        ):
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence v2 index row-key order token mismatch",
            )

        evidence = self._load_canonical_evidence(
            partition_key=partition_key,
            evidence_id=indexed.evidence_id,
        )

        if str(evidence.evidence_id) != indexed.evidence_id:
            raise CausalEvidencePersistenceIntegrityError(
                "canonical causal evidence id does not match index reference",
            )

        if evidence.recorded_at != indexed.recorded_at:
            raise CausalEvidencePersistenceIntegrityError(
                "canonical causal evidence recorded_at does not match v2 index metadata",
            )

        validate_scope(evidence)

        return evidence

    def _load_canonical_evidence(
        self,
        *,
        partition_key: str,
        evidence_id: str,
    ) -> PlatformCausalEvidence:

        record = self._document_store.get(partition_key, _record_row_key(evidence_id))

        if record is None:
            raise CausalEvidencePersistenceIntegrityError(
                "canonical causal evidence record missing for index",
            )

        return decode_causal_evidence_record(dict(record.data))

    def _ensure_indexes(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> None:
        exec_prefix = execution_index_v1_row_key_prefix(
            task_id=evidence.target.task_id,
            run_id=evidence.target.run_id,
        )
        transport_prefix = transport_index_v1_row_key_prefix(
            provider=evidence.source.provider,
            transport_task_id=evidence.source.task_id,
        )
        had_exec_rows = self._scope_has_any_index_rows(
            partition_key=partition_key,
            row_key_prefix=exec_prefix,
        )
        had_transport_rows = self._scope_has_any_index_rows(
            partition_key=partition_key,
            row_key_prefix=transport_prefix,
        )
        exec_v2 = self._execution_v2_index_document(
            evidence=evidence,
            partition_key=partition_key,
        )
        transport_v2 = self._transport_v2_index_document(
            evidence=evidence,
            partition_key=partition_key,
        )
        for document in (exec_v2, transport_v2):
            if not self._document_store.put_if_absent(document):
                self._verify_v2_index_document(document, evidence)

        self._advance_high_water(
            partition_key=partition_key,
            marker_row_key=_high_water_exec_row_key(
                task_id=evidence.target.task_id,
                run_id=evidence.target.run_id,
            ),
            row_key=exec_v2.row_key,
        )
        self._advance_high_water(
            partition_key=partition_key,
            marker_row_key=_high_water_transport_row_key(
                provider=evidence.source.provider,
                transport_task_id=evidence.source.task_id,
            ),
            row_key=transport_v2.row_key,
        )
        if not had_exec_rows:
            self._mark_execution_v2_ready(
                partition_key=partition_key,
                task_id=evidence.target.task_id,
                run_id=evidence.target.run_id,
            )
        if not had_transport_rows:
            self._mark_transport_v2_ready(
                partition_key=partition_key,
                provider=evidence.source.provider,
                transport_task_id=evidence.source.task_id,
            )

    def _put_v2_execution_index(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> None:

        document = self._execution_v2_index_document(
            evidence=evidence,
            partition_key=partition_key,
        )

        if not self._document_store.put_if_absent(document):
            self._verify_v2_index_document(document, evidence)

        self._advance_high_water(
            partition_key=partition_key,
            marker_row_key=_high_water_exec_row_key(
                task_id=evidence.target.task_id,
                run_id=evidence.target.run_id,
            ),
            row_key=document.row_key,
        )

    def _put_v2_transport_index(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> None:

        document = self._transport_v2_index_document(
            evidence=evidence,
            partition_key=partition_key,
        )

        if not self._document_store.put_if_absent(document):
            self._verify_v2_index_document(document, evidence)

        self._advance_high_water(
            partition_key=partition_key,
            marker_row_key=_high_water_transport_row_key(
                provider=evidence.source.provider,
                transport_task_id=evidence.source.task_id,
            ),
            row_key=document.row_key,
        )

    def _execution_v1_index_document(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> DocumentRecord:

        return DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v1_row_key(
                task_id=evidence.target.task_id,
                run_id=evidence.target.run_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=encode_causal_evidence_index_v1(str(evidence.evidence_id)),
        )

    def _transport_v1_index_document(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> DocumentRecord:

        return DocumentRecord(
            partition_key=partition_key,
            row_key=transport_index_v1_row_key(
                provider=evidence.source.provider,
                transport_task_id=evidence.source.task_id,
                evidence_id=str(evidence.evidence_id),
            ),
            data=encode_causal_evidence_index_v1(str(evidence.evidence_id)),
        )

    def _execution_v2_index_document(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> DocumentRecord:

        return DocumentRecord(
            partition_key=partition_key,
            row_key=execution_index_v2_row_key_from_evidence(evidence),
            data=encode_causal_evidence_index_v2(evidence),
        )

    def _transport_v2_index_document(
        self,
        *,
        evidence: PlatformCausalEvidence,
        partition_key: str,
    ) -> DocumentRecord:

        return DocumentRecord(
            partition_key=partition_key,
            row_key=transport_index_v2_row_key_from_evidence(evidence),
            data=encode_causal_evidence_index_v2(evidence),
        )

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

        indexed_id = self._decode_index_ref(dict(existing.data))

        if indexed_id != str(incoming.evidence_id):
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence index conflicts with expected evidence_id",
            )

    def _verify_v2_index_document(
        self,
        document: DocumentRecord,
        incoming: PlatformCausalEvidence,
    ) -> None:

        existing = self._document_store.get(document.partition_key, document.row_key)

        if existing is None:
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence v2 index verification failed",
            )

        try:
            indexed = decode_causal_evidence_index_v2(dict(existing.data))

        except ValueError as exc:
            raise CausalEvidencePersistenceIntegrityError(
                "invalid causal evidence v2 index",
            ) from exc

        if indexed.evidence_id != str(incoming.evidence_id):
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence v2 index conflicts with expected evidence_id",
            )

        if indexed.recorded_at != incoming.recorded_at:
            raise CausalEvidencePersistenceIntegrityError(
                "causal evidence v2 index conflicts with expected recorded_at",
            )

    def _decode_index_ref(self, data: dict[str, str]) -> str:

        schema_version = data.get("schema_version")

        if schema_version == "intergrax.causal_evidence.index.v1":
            return decode_causal_evidence_index_v1(data).evidence_id

        if schema_version == "intergrax.causal_evidence.index.v2":
            return decode_causal_evidence_index_v2(data).evidence_id

        if schema_version == _LEGACY_RECORD_SCHEMA:
            return str(decode_causal_evidence_record(data).evidence_id)

        raise CausalEvidencePersistenceIntegrityError(
            "unsupported causal evidence index schema",
        )

    def _scope_has_any_index_rows(
        self,
        *,
        partition_key: str,
        row_key_prefix: str,
    ) -> bool:
        page = self._document_store.query(
            partition_key,
            limit=1,
            row_key_prefix=row_key_prefix,
        )
        return any(
            not document.row_key.startswith("meta:") for document in page.documents
        )

    def _is_execution_v2_ready(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> bool:

        marker = self._document_store.get(
            partition_key,
            _v2_ready_exec_row_key(task_id=task_id, run_id=run_id),
        )

        return marker is not None

    def _is_transport_v2_ready(
        self,
        *,
        partition_key: str,
        provider: str,
        transport_task_id: str,
    ) -> bool:

        marker = self._document_store.get(
            partition_key,
            _v2_ready_transport_row_key(
                provider=provider,
                transport_task_id=transport_task_id,
            ),
        )

        return marker is not None

    def _mark_execution_v2_ready(
        self,
        *,
        partition_key: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> None:

        self._document_store.put_if_absent(
            DocumentRecord(
                partition_key=partition_key,
                row_key=_v2_ready_exec_row_key(task_id=task_id, run_id=run_id),
                data=_encode_scope_marker(),
            ),
        )

    def _mark_transport_v2_ready(
        self,
        *,
        partition_key: str,
        provider: str,
        transport_task_id: str,
    ) -> None:

        self._document_store.put_if_absent(
            DocumentRecord(
                partition_key=partition_key,
                row_key=_v2_ready_transport_row_key(
                    provider=provider,
                    transport_task_id=transport_task_id,
                ),
                data=_encode_scope_marker(),
            ),
        )

    def _read_high_water(
        self, *, partition_key: str, marker_row_key: str
    ) -> str | None:

        marker = self._document_store.get(partition_key, marker_row_key)

        if marker is None:
            return None

        return _decode_scope_row_key(dict(marker.data))

    def _advance_high_water(
        self,
        *,
        partition_key: str,
        marker_row_key: str,
        row_key: str,
    ) -> None:

        current = self._read_high_water(
            partition_key=partition_key, marker_row_key=marker_row_key
        )

        if current is not None and current >= row_key:
            return

        self._document_store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=marker_row_key,
                data=_encode_scope_marker(row_key=row_key),
            ),
        )

    def _read_reconcile_cursor(
        self,
        partition_key: str,
        reconcile_row_key: str,
    ) -> str | None:

        marker = self._document_store.get(partition_key, reconcile_row_key)

        if marker is None:
            return None

        return _decode_scope_row_key(dict(marker.data))

    @staticmethod
    def _validate_execution_scope(
        evidence: PlatformCausalEvidence,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> None:

        if (
            evidence.tenant_id != tenant_id
            or evidence.target.task_id != task_id
            or evidence.target.run_id != run_id
        ):
            raise CausalEvidencePersistenceIntegrityError(
                "canonical causal evidence does not match execution index scope",
            )

    @staticmethod
    def _validate_transport_scope(
        evidence: PlatformCausalEvidence,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
    ) -> None:

        if (
            evidence.tenant_id != tenant_id
            or evidence.source.provider != provider
            or evidence.source.task_id != transport_task_id
        ):
            raise CausalEvidencePersistenceIntegrityError(
                "canonical causal evidence does not match transport index scope",
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
