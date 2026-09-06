"""Parquet encode/decode for relational data pack records."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    source_ref_from_columns,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)


def write_relational_parquet(path: Path, records: Sequence[RelationalDataPackRecord]) -> None:
    if not records:
        raise VpiDataPackIntegrityError("cannot write empty relational shard")
    table = pa.table(
        {
            "global_row_index": pa.array(
                [record.global_row_index for record in records],
                type=pa.int64(),
            ),
            "catalog_id": pa.array(
                [record.source_ref.catalog_id for record in records],
                type=pa.string(),
            ),
            "offer_id": pa.array(
                [record.source_ref.offer_id.value for record in records],
                type=pa.string(),
            ),
            "source_revision": pa.array(
                [record.source_ref.source_revision for record in records],
                type=pa.string(),
            ),
            "record_json": pa.array(
                [record.record_json for record in records],
                type=pa.string(),
            ),
            "derivation_version": pa.array(
                [record.derivation_version for record in records],
                type=pa.string(),
            ),
            "semantic_text": pa.array(
                [record.semantic_text for record in records],
                type=pa.string(),
            ),
            "semantic_text_hash": pa.array(
                [record.semantic_text_hash for record in records],
                type=pa.string(),
            ),
            "title": pa.array([record.title for record in records], type=pa.string()),
            "brand": pa.array([record.brand for record in records], type=pa.string()),
            "category": pa.array([record.category for record in records], type=pa.string()),
            "description": pa.array(
                [record.description for record in records],
                type=pa.string(),
            ),
            "has_identifiers": pa.array(
                [record.has_identifiers for record in records],
                type=pa.bool_(),
            ),
            "has_spec_table": pa.array(
                [record.has_spec_table for record in records],
                type=pa.bool_(),
            ),
            "has_structured_attributes": pa.array(
                [record.has_structured_attributes for record in records],
                type=pa.bool_(),
            ),
        }
    )
    try:
        pq.write_table(table, path)
    except OSError as exc:
        raise VpiDataPackIntegrityError(f"failed to write relational parquet: {path}") from exc


def read_relational_parquet(path: Path) -> tuple[RelationalDataPackRecord, ...]:
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as exc:
        raise VpiDataPackIntegrityError(f"failed to read relational parquet: {path}") from exc
    required = (
        "global_row_index",
        "catalog_id",
        "offer_id",
        "source_revision",
        "record_json",
        "derivation_version",
        "semantic_text",
        "semantic_text_hash",
        "title",
        "brand",
        "category",
        "description",
        "has_identifiers",
        "has_spec_table",
        "has_structured_attributes",
    )
    for column_name in required:
        if column_name not in table.column_names:
            raise VpiDataPackIntegrityError(f"relational shard missing column: {column_name}")

    records: list[RelationalDataPackRecord] = []
    for row_index in range(table.num_rows):
        source_revision_raw = table.column("source_revision")[row_index].as_py()
        source_revision = str(source_revision_raw) if source_revision_raw is not None else None
        records.append(
            RelationalDataPackRecord(
                global_row_index=int(table.column("global_row_index")[row_index].as_py()),
                source_ref=source_ref_from_columns(
                    catalog_id=str(table.column("catalog_id")[row_index].as_py()),
                    offer_id=str(table.column("offer_id")[row_index].as_py()),
                    source_revision=source_revision,
                ),
                record_json=str(table.column("record_json")[row_index].as_py()),
                derivation_version=str(table.column("derivation_version")[row_index].as_py()),
                semantic_text=str(table.column("semantic_text")[row_index].as_py()),
                semantic_text_hash=str(table.column("semantic_text_hash")[row_index].as_py()),
                title=table.column("title")[row_index].as_py(),
                brand=table.column("brand")[row_index].as_py(),
                category=table.column("category")[row_index].as_py(),
                description=table.column("description")[row_index].as_py(),
                has_identifiers=bool(table.column("has_identifiers")[row_index].as_py()),
                has_spec_table=bool(table.column("has_spec_table")[row_index].as_py()),
                has_structured_attributes=bool(
                    table.column("has_structured_attributes")[row_index].as_py()
                ),
            )
        )
    return tuple(records)
