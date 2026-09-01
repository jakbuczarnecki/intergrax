# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB collection client — collection injected from ``opens.py`` only."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Optional

from intergrax.integrations._shared.document_store_query_support import (
    build_mongo_data_field,
    build_mongo_equality_filter,
    build_mongo_keyset_filter,
    decode_v2_sort_values,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentQueryCursorCodec,
    DocumentRecord,
    normalize_document_data_equalities,
    normalize_document_data_sort,
    validate_document_query_limit,
)
from intergrax.integrations.contracts.partition_atomic_document_store import (
    PartitionAtomicBatch,
    PartitionAtomicBatchResult,
    PartitionPutIfAbsentOnCreated,
    PartitionReplaceIfMatchOnCreated,
    validate_partition_atomic_batch,
)
from intergrax.integrations.providers.document_store.mongodb.config import MongoDBIntegrationConfig


class PartitionAtomicBatchConflictError(RuntimeError):
    """Raised when an on-created operation cannot commit inside a partition batch."""


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _is_expired(expires_at: object) -> bool:
    if expires_at is None:
        return False
    if not isinstance(expires_at, datetime):
        return False
    compare_at = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)
    return compare_at <= _utcnow()


def _document_filter(partition_key: str, row_key: str) -> dict[str, str]:
    return {"partition_key": partition_key, "row_key": row_key}


def _row_from_doc(doc: Mapping[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        partition_key=str(doc.get("partition_key", "")),
        row_key=str(doc.get("row_key", "")),
        data=dict(doc.get("data") or {}),
    )


def _payload_from_document(document: DocumentRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "partition_key": document.partition_key,
        "row_key": document.row_key,
        "data": dict(document.data),
    }
    if document.ttl_seconds is not None and document.ttl_seconds > 0:
        payload["expires_at"] = _utcnow() + timedelta(seconds=int(document.ttl_seconds))
    else:
        payload["expires_at"] = None
    return payload


def _require_matching_keys(*, expected: DocumentRecord, replacement: DocumentRecord) -> None:
    if (
        expected.partition_key != replacement.partition_key
        or expected.row_key != replacement.row_key
    ):
        raise ValueError(
            "replace_if_match requires expected and replacement to share "
            "partition_key and row_key"
        )


def _match_filter(expected: DocumentRecord) -> dict[str, Any]:
    return {
        "partition_key": expected.partition_key,
        "row_key": expected.row_key,
        "data": dict(expected.data),
    }


class MongoCollectionClient:
    """Minimal MongoDB client for partition-scoped document CRUD."""

    def __init__(
        self,
        config: MongoDBIntegrationConfig,
        *,
        collection: Any,
        client: Any | None = None,
        is_duplicate_key_error: Callable[[BaseException], bool] | None = None,
    ) -> None:
        if not config.uri:
            raise IntegrationConfigurationError(
                "MongoDB uri is required (INTERGRAX_MONGODB_URI)"
            )
        self._config = config
        self._collection = collection
        self._client = client
        self._is_duplicate_key_error = is_duplicate_key_error or (lambda _exc: False)

    @property
    def config(self) -> MongoDBIntegrationConfig:
        return self._config

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        doc = self._collection.find_one(_document_filter(partition_key, row_key))
        if doc is None:
            return None
        if _is_expired(doc.get("expires_at")):
            self._collection.delete_one(_document_filter(partition_key, row_key))
            return None
        return _row_from_doc(doc)

    def put(self, document: DocumentRecord) -> None:
        payload = _payload_from_document(document)
        self._collection.replace_one(
            _document_filter(document.partition_key, document.row_key),
            payload,
            upsert=True,
        )

    def delete(self, partition_key: str, row_key: str) -> None:
        self._collection.delete_one(_document_filter(partition_key, row_key))

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
        after_row_key: Optional[str] = None,
        row_key_upper_bound: Optional[str] = None,
        data_equalities: Sequence[DocumentDataEquality] = (),
        sort: Sequence[DocumentDataSort] = (),
        cursor_codec: DocumentQueryCursorCodec | None = None,
        cursor: Optional[str] = None,
    ) -> list[DocumentRecord]:
        bounded_limit = validate_document_query_limit(limit)
        normalized_equalities = normalize_document_data_equalities(data_equalities)
        normalized_sort = normalize_document_data_sort(sort)

        if normalized_sort and cursor is not None:
            if cursor_codec is None:
                raise ValueError("document_store_cursor_invalid")
            payload = cursor_codec.decode_v2(
                cursor,
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                row_key_upper_bound=row_key_upper_bound,
                data_equalities=normalized_equalities,
                sort=normalized_sort,
            )
            keyset_filter = build_mongo_keyset_filter(
                normalized_sort,
                decode_v2_sort_values(payload),
            )
        elif cursor is not None and cursor_codec is not None and not normalized_sort:
            after_row_key = cursor_codec.decode_v2(
                cursor,
                partition_key=partition_key,
                row_key_prefix=row_key_prefix,
                row_key_upper_bound=row_key_upper_bound,
                data_equalities=normalized_equalities,
                sort=normalized_sort,
            ).last_row_key
            keyset_filter = None
        else:
            keyset_filter = None

        query_filter: dict[str, Any] = {"partition_key": partition_key}
        row_key_filter: dict[str, Any] = {}
        if row_key_prefix:
            row_key_filter["$regex"] = f"^{row_key_prefix}"
        if after_row_key:
            row_key_filter["$gt"] = after_row_key
        if row_key_upper_bound is not None:
            row_key_filter["$lte"] = row_key_upper_bound
        if row_key_filter:
            query_filter["row_key"] = row_key_filter
        query_filter.update(build_mongo_equality_filter(normalized_equalities))
        if keyset_filter is not None:
            query_filter.update(keyset_filter)

        mongo_sort: list[tuple[str, int]] = []
        for spec in normalized_sort:
            direction = -1 if spec.direction == "desc" else 1
            mongo_sort.append((build_mongo_data_field(spec.path), direction))

        mongo_cursor = self._collection.find(query_filter)
        if mongo_sort:
            mongo_cursor = mongo_cursor.sort(mongo_sort)
        else:
            mongo_cursor = mongo_cursor.sort("row_key", 1)
        mongo_cursor = mongo_cursor.limit(bounded_limit)

        documents: list[DocumentRecord] = []
        for doc in mongo_cursor:
            if _is_expired(doc.get("expires_at")):
                self._collection.delete_one(
                    _document_filter(str(doc.get("partition_key", "")), str(doc.get("row_key", "")))
                )
                continue
            documents.append(_row_from_doc(doc))
        return documents

    def put_if_absent(self, document: DocumentRecord) -> bool:
        payload = _payload_from_document(document)
        try:
            result = self._collection.update_one(
                _document_filter(document.partition_key, document.row_key),
                {"$setOnInsert": payload},
                upsert=True,
            )
        except Exception as exc:
            if self._is_duplicate_key_error(exc):
                return False
            raise
        return result.upserted_id is not None

    def replace_if_match(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
    ) -> bool:
        _require_matching_keys(expected=expected, replacement=replacement)
        result = self._collection.replace_one(
            _match_filter(expected),
            _payload_from_document(replacement),
            upsert=False,
        )
        return int(result.matched_count) == 1

    def delete_if_match(self, *, expected: DocumentRecord) -> bool:
        result = self._collection.delete_one(_match_filter(expected))
        return int(result.deleted_count) == 1

    def supports_partition_atomic_batch(self) -> bool:
        if self._client is None:
            return False
        try:
            hello = self._client.admin.command("hello")
        except Exception:
            return False
        set_name = hello.get("setName")
        return isinstance(set_name, str) and bool(set_name)

    def execute_partition_atomic_batch(
        self,
        batch: PartitionAtomicBatch,
    ) -> PartitionAtomicBatchResult:
        validated = validate_partition_atomic_batch(batch)
        if self._client is None:
            raise IntegrationConfigurationError(
                "MongoDB partition atomic batch requires an open pymongo client",
            )
        if not self.supports_partition_atomic_batch():
            raise IntegrationConfigurationError(
                "MongoDB partition atomic batch requires a replica-set deployment",
            )

        def _callback(session: object) -> PartitionAtomicBatchResult:
            primary_created = self._put_if_absent_in_session(
                validated.primary_put_if_absent,
                session=session,
            )
            if primary_created:
                for op in validated.on_created_ops:
                    if isinstance(op, PartitionPutIfAbsentOnCreated):
                        if not self._put_if_absent_in_session(
                            op.document,
                            session=session,
                        ):
                            raise PartitionAtomicBatchConflictError(
                                "partition_atomic_batch_on_created_conflict",
                            )
                    elif isinstance(op, PartitionReplaceIfMatchOnCreated):
                        if not self._replace_if_match_in_session(
                            expected=op.expected,
                            replacement=op.replacement,
                            session=session,
                        ):
                            raise PartitionAtomicBatchConflictError(
                                "partition_atomic_batch_on_created_stale",
                            )
                    else:
                        raise TypeError("partition_atomic_batch_on_created_op_invalid")
            return PartitionAtomicBatchResult(primary_created=primary_created)

        try:
            with self._client.start_session() as session:
                return session.with_transaction(_callback)
        except PartitionAtomicBatchConflictError:
            raise
        except Exception as exc:
            if self._is_retryable_partition_batch_error(exc):
                raise PartitionAtomicBatchConflictError(
                    "partition_atomic_batch_transient",
                ) from exc
            raise

    @staticmethod
    def _is_retryable_partition_batch_error(exc: BaseException) -> bool:
        try:
            from pymongo.errors import OperationFailure, PyMongoError
        except ImportError:
            return False
        if isinstance(exc, OperationFailure):
            labels = getattr(exc, "details", {}) or {}
            error_labels = labels.get("errorLabels") or getattr(exc, "_error_labels", ())
            if "TransientTransactionError" in error_labels:
                return True
            if "UnknownTransactionCommitResult" in error_labels:
                return True
            return int(getattr(exc, "code", 0)) in {112, 251}
        return isinstance(exc, PyMongoError)

    def _put_if_absent_in_session(
        self,
        document: DocumentRecord,
        *,
        session: object,
    ) -> bool:
        payload = _payload_from_document(document)
        try:
            result = self._collection.update_one(
                _document_filter(document.partition_key, document.row_key),
                {"$setOnInsert": payload},
                upsert=True,
                session=session,
            )
        except Exception as exc:
            if self._is_duplicate_key_error(exc):
                return False
            raise
        return result.upserted_id is not None

    def _replace_if_match_in_session(
        self,
        *,
        expected: DocumentRecord,
        replacement: DocumentRecord,
        session: object,
    ) -> bool:
        _require_matching_keys(expected=expected, replacement=replacement)
        result = self._collection.replace_one(
            _match_filter(expected),
            _payload_from_document(replacement),
            upsert=False,
            session=session,
        )
        return int(result.matched_count) == 1

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
