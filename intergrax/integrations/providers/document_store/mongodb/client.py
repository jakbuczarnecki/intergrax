# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""MongoDB collection client — collection injected from ``opens.py`` only."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
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


class MongoCollectionClient:
    """Minimal MongoDB client for partition-scoped document CRUD."""

    def __init__(
        self,
        config: MongoDBIntegrationConfig,
        *,
        collection: Any,
        client: Any | None = None,
    ) -> None:
        if not config.uri:
            raise IntegrationConfigurationError(
                "MongoDB uri is required (INTERGRAX_MONGODB_URI)"
            )
        self._config = config
        self._collection = collection
        self._client = client

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
        payload: dict[str, Any] = {
            "partition_key": document.partition_key,
            "row_key": document.row_key,
            "data": dict(document.data),
        }
        if document.ttl_seconds is not None and document.ttl_seconds > 0:
            payload["expires_at"] = _utcnow() + timedelta(seconds=int(document.ttl_seconds))
        else:
            payload["expires_at"] = None
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
    ) -> DocumentQueryResult:
        bounded_limit = max(1, int(limit))
        query_filter: dict[str, Any] = {"partition_key": partition_key}
        if row_key_prefix:
            query_filter["row_key"] = {"$regex": f"^{row_key_prefix}"}
        cursor = self._collection.find(query_filter).sort("row_key", 1).limit(bounded_limit)
        documents: list[DocumentRecord] = []
        for doc in cursor:
            if _is_expired(doc.get("expires_at")):
                self._collection.delete_one(
                    _document_filter(str(doc.get("partition_key", "")), str(doc.get("row_key", "")))
                )
                continue
            documents.append(_row_from_doc(doc))
        return DocumentQueryResult(documents=documents, total=len(documents))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
