"""Adapter-owned typed row contract for PostgreSQL structured attribute reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapOperationError,
)

_PostgreSqlFetchedValue = str | int | None


@dataclass(frozen=True, slots=True)
class StoredStructuredCandidateRow:
    """Immutable aggregated structured search row after PostgreSQL boundary conversion."""

    catalog_id: str
    offer_id: str
    source_revision_norm: str
    source_revision: str | None
    matched_constraint_count: int


def _require_str(
    row: Mapping[str, _PostgreSqlFetchedValue],
    field: str,
) -> str:
    if field not in row:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL structured search row: missing {field}"
        )
    value = row[field]
    if not isinstance(value, str):
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL structured search row: {field} must be str"
        )
    return value


def _optional_str(
    row: Mapping[str, _PostgreSqlFetchedValue],
    field: str,
) -> str | None:
    if field not in row:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL structured search row: missing {field}"
        )
    value = row[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL structured search row: {field} must be str or null"
        )
    return value


def _require_int(
    row: Mapping[str, _PostgreSqlFetchedValue],
    field: str,
) -> int:
    if field not in row:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL structured search row: missing {field}"
        )
    value = row[field]
    if type(value) is not int:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL structured search row: {field} must be int"
        )
    return value


def stored_structured_candidate_row_from_fetched_row(
    row: Mapping[str, _PostgreSqlFetchedValue],
) -> StoredStructuredCandidateRow:
    """Convert a platform PostgreSQL fetched row into a typed structured candidate row."""
    return StoredStructuredCandidateRow(
        catalog_id=_require_str(row, "catalog_id"),
        offer_id=_require_str(row, "offer_id"),
        source_revision_norm=_require_str(row, "source_revision_norm"),
        source_revision=_optional_str(row, "source_revision"),
        matched_constraint_count=_require_int(row, "matched_constraint_count"),
    )
