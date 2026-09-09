"""Adapter-owned typed row contract for PostgreSQL identifier index reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapOperationError,
)

_PostgreSqlFetchedValue = str | None


@dataclass(frozen=True, slots=True)
class StoredIdentifierRow:
    """Immutable stored identifier index row after PostgreSQL boundary conversion."""

    catalog_id: str
    offer_id: str
    source_revision_norm: str
    source_revision: str | None
    identifier_type: ProductIdentifierType
    source_value: str
    normalized_value: str
    source_field: str


def _require_str(
    row: Mapping[str, _PostgreSqlFetchedValue],
    field: str,
) -> str:
    if field not in row:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL identifier row: missing {field}"
        )
    value = row[field]
    if not isinstance(value, str):
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL identifier row: {field} must be str"
        )
    return value


def _optional_str(
    row: Mapping[str, _PostgreSqlFetchedValue],
    field: str,
) -> str | None:
    if field not in row:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL identifier row: missing {field}"
        )
    value = row[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL identifier row: {field} must be str or null"
        )
    return value


def stored_identifier_row_from_fetched_row(
    row: Mapping[str, _PostgreSqlFetchedValue],
) -> StoredIdentifierRow:
    """Convert a platform PostgreSQL fetched row into a typed identifier contract."""
    identifier_type_raw = _require_str(row, "identifier_type")
    try:
        identifier_type = ProductIdentifierType(identifier_type_raw)
    except ValueError as exc:
        raise PostgreSqlBootstrapOperationError(
            f"malformed PostgreSQL identifier row: unknown identifier_type {identifier_type_raw!r}"
        ) from exc
    return StoredIdentifierRow(
        catalog_id=_require_str(row, "catalog_id"),
        offer_id=_require_str(row, "offer_id"),
        source_revision_norm=_require_str(row, "source_revision_norm"),
        source_revision=_optional_str(row, "source_revision"),
        identifier_type=identifier_type,
        source_value=_require_str(row, "source_value"),
        normalized_value=_require_str(row, "normalized_value"),
        source_field=_require_str(row, "source_field"),
    )
