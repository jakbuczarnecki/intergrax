"""Load arena sample records from the VPI dataset."""

from __future__ import annotations

from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    VpiEmbeddingMaterializationConfig,
)
from platform_proofs.scenarios.verified_product_identification.ingest.pipeline.derive_batch import (
    build_catalog_ingest_batch,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    iter_dataset_rows,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ARENA_SAMPLE_SCAN_ROW_LIMIT,
    ArenaSampleRecord,
    build_arena_sample_manifest,
    derive_strata_tags,
    select_arena_sample_records,
)


def load_arena_sample_records(
    config: VpiEmbeddingMaterializationConfig,
    *,
    scan_row_limit: int = ARENA_SAMPLE_SCAN_ROW_LIMIT,
    target_size: int = 1000,
) -> tuple[ArenaSampleRecord, ...]:
    candidates: list[ArenaSampleRecord] = []
    for _, rows in iter_dataset_rows(
        config.dataset_path,
        batch_size=config.source_read_batch_size,
        start_row_index=0,
        start_batch_ordinal=0,
        max_records=scan_row_limit,
    ):
        catalog_batch = build_catalog_ingest_batch(
            batch_ordinal=0,
            rows=rows,
            catalog_id=config.catalog_id,
            source_revision=config.source_revision,
        )
        for record in catalog_batch.records:
            source_offer = parse_wdc_source_offer_json(record.record_json)
            candidates.append(
                ArenaSampleRecord(
                    offer_id=source_offer.offer_id,
                    global_row_index=record.global_row_index,
                    semantic_text=record.representation.semantic.semantic_text,
                    source_offer=source_offer,
                    strata_tags=derive_strata_tags(source_offer),
                )
            )
    return select_arena_sample_records(tuple(candidates), target_size=target_size)


def load_arena_sample_manifest_from_config(
    config: VpiEmbeddingMaterializationConfig,
    *,
    scan_row_limit: int = ARENA_SAMPLE_SCAN_ROW_LIMIT,
    target_size: int = 1000,
) -> object:
    records = load_arena_sample_records(
        config,
        scan_row_limit=scan_row_limit,
        target_size=target_size,
    )
    return build_arena_sample_manifest(records, scan_row_limit=scan_row_limit)
