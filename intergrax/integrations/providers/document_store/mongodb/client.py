# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB collection client — collection injected from ``opens.py`` only."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentRecord, validate_document_query_limit
from intergrax.integrations.providers.document_store.mongodb.config import MongoDBIntegrationConfig


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
    ) -> list[DocumentRecord]:
        bounded_limit = validate_document_query_limit(limit)
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
        mongo_cursor = (
            self._collection.find(query_filter).sort("row_key", 1).limit(bounded_limit)
        )
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

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
