# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cassandra CQL client — session injected from ``opens.py`` only."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentQueryResult, DocumentRecord
from intergrax.integrations.providers.document_store.cassandra.config import CassandraIntegrationConfig


def _decode_payload(raw: object) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IntegrationConfigurationError("Cassandra payload is not valid JSON") from exc
        if isinstance(parsed, dict):
            return parsed
        raise IntegrationConfigurationError("Cassandra payload must decode to a JSON object")
    raise IntegrationConfigurationError("Unexpected Cassandra payload type")


def _encode_payload(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), separators=(",", ":"), sort_keys=True)


class CassandraCqlClient:
    """Minimal CQL client for partition-scoped document CRUD."""

    def __init__(
        self,
        config: CassandraIntegrationConfig,
        *,
        session: Any,
    ) -> None:
        if not config.contact_points_list():
            raise IntegrationConfigurationError(
                "Cassandra contact_points are required (INTERGRAX_CASSANDRA_CONTACT_POINTS)"
            )
        if not config.keyspace:
            raise IntegrationConfigurationError(
                "Cassandra keyspace is required (INTERGRAX_CASSANDRA_KEYSPACE)"
            )
        self._config = config
        self._session = session
        table = config.qualified_table()
        self._select_one = session.prepare(
            f"SELECT payload FROM {table} WHERE partition_key = ? AND row_key = ?"
        )
        self._insert = session.prepare(
            f"INSERT INTO {table} (partition_key, row_key, payload) VALUES (?, ?, ?)"
        )
        self._insert_ttl = session.prepare(
            f"INSERT INTO {table} (partition_key, row_key, payload) VALUES (?, ?, ?) USING TTL ?"
        )
        self._delete = session.prepare(
            f"DELETE FROM {table} WHERE partition_key = ? AND row_key = ?"
        )
        self._select_partition = session.prepare(
            f"SELECT row_key, payload FROM {table} WHERE partition_key = ? LIMIT ?"
        )
        self._select_partition_range = session.prepare(
            f"SELECT row_key, payload FROM {table} "
            f"WHERE partition_key = ? AND row_key >= ? AND row_key < ? LIMIT ?"
        )

    @property
    def config(self) -> CassandraIntegrationConfig:
        return self._config

    def get(self, partition_key: str, row_key: str) -> Optional[DocumentRecord]:
        row = self._session.execute(self._select_one, (partition_key, row_key)).one()
        if row is None:
            return None
        payload = getattr(row, "payload", None)
        return DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data=_decode_payload(payload),
        )

    def put(self, document: DocumentRecord) -> None:
        payload = _encode_payload(document.data)
        if document.ttl_seconds is not None and document.ttl_seconds > 0:
            self._session.execute(
                self._insert_ttl,
                (
                    document.partition_key,
                    document.row_key,
                    payload,
                    int(document.ttl_seconds),
                ),
            )
            return
        self._session.execute(
            self._insert,
            (document.partition_key, document.row_key, payload),
        )

    def delete(self, partition_key: str, row_key: str) -> None:
        self._session.execute(self._delete, (partition_key, row_key))

    def query(
        self,
        partition_key: str,
        *,
        limit: int = 100,
        row_key_prefix: Optional[str] = None,
    ) -> DocumentQueryResult:
        bounded_limit = max(1, int(limit))
        if row_key_prefix:
            end_key = f"{row_key_prefix}\uffff"
            rows = self._session.execute(
                self._select_partition_range,
                (partition_key, row_key_prefix, end_key, bounded_limit),
            )
        else:
            rows = self._session.execute(
                self._select_partition,
                (partition_key, bounded_limit),
            )
        documents = [
            DocumentRecord(
                partition_key=partition_key,
                row_key=str(getattr(row, "row_key", "")),
                data=_decode_payload(getattr(row, "payload", None)),
            )
            for row in rows
        ]
        return DocumentQueryResult(documents=documents, total=len(documents))

    def shutdown(self) -> None:
        cluster = getattr(self._session, "cluster", None)
        self._session.shutdown()
        if cluster is not None:
            cluster.shutdown()
