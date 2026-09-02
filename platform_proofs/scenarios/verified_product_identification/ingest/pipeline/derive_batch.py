"""Derive catalog/search ingest records from streaming dataset rows."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.errors import (
    VpiBootstrapDataError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    CatalogIngestBatch,
    CatalogIngestRecord,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    DatasetRow,
)


def build_catalog_ingest_batch(
    *,
    batch_ordinal: int,
    rows: tuple[DatasetRow, ...],
    catalog_id: str,
    source_revision: str | None,
) -> CatalogIngestBatch:
    records: list[CatalogIngestRecord] = []
    for row in rows:
        try:
            source_offer = parse_wdc_source_offer_json(row.record_json)
            source_ref = build_source_record_ref(
                source_offer,
                catalog_id=catalog_id,
                source_revision=source_revision,
            )
            representation = derive_search_representation(
                source_offer,
                source_ref=source_ref,
            )
        except (ValueError, TypeError) as exc:
            raise VpiBootstrapDataError(
                f"failed to derive search representation at row {row.global_row_index}"
            ) from exc
        records.append(
            CatalogIngestRecord(
                global_row_index=row.global_row_index,
                record_json=row.record_json,
                source_offer=source_offer,
                representation=representation,
            )
        )
    return CatalogIngestBatch(batch_ordinal=batch_ordinal, records=tuple(records))
