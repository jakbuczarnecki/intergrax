"""Adapter-owned typed row contract for PostgreSQL relational bootstrap reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapOperationError,
)

_PostgreSqlFetchedValue = str | int | None


@dataclass(frozen=True, slots=True)
class StoredRelationalRow:
    """Immutable stored relational row used after PostgreSQL boundary conversion."""

    catalog_id: str
    offer_id: str
    source_revision_norm: str
    source_revision: str | None
    global_row_index: int
    record_json: str
    semantic_text: str
    semantic_text_hash: str
    derivation_version: str


def _require_str(
    row: Mapping[str, _PostgreSqlFetchedValue],
    field: str,
) -> str:
    if field not in row:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL row: missing {field}"
        )
    value = row[field]
    if not isinstance(value, str):
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL row: {field} must be str"
        )
    return value


def _require_int(
    row: Mapping[str, _PostgreSqlFetchedValue],
    field: str,
) -> int:
    if field not in row:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL row: missing {field}"
        )
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL row: {field} must be int"
        )
    return value


def _optional_str(
    row: Mapping[str, _PostgreSqlFetchedValue],
    field: str,
) -> str | None:
    if field not in row:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL row: missing {field}"
        )
    value = row[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL row: {field} must be str or null"
        )
    return value


def stored_relational_row_from_fetched_row(
    row: Mapping[str, _PostgreSqlFetchedValue],
) -> StoredRelationalRow:
    """Convert a platform PostgreSQL fetched row into an adapter-owned typed contract."""
    return StoredRelationalRow(
        catalog_id=_require_str(row, "catalog_id"),
        offer_id=_require_str(row, "offer_id"),
        source_revision_norm=_require_str(row, "source_revision_norm"),
        source_revision=_optional_str(row, "source_revision"),
        global_row_index=_require_int(row, "global_row_index"),
        record_json=_require_str(row, "record_json"),
        semantic_text=_require_str(row, "semantic_text"),
        semantic_text_hash=_require_str(row, "semantic_text_hash"),
        derivation_version=_require_str(row, "derivation_version"),
    )
