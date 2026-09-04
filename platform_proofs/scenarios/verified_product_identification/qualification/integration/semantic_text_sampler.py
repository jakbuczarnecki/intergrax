"""Sample real VPI semantic texts for qualification microbenchmarks."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    VpiEmbeddingMaterializationConfig,
)
from platform_proofs.scenarios.verified_product_identification.ingest.pipeline.derive_batch import (
    build_catalog_ingest_batch,
)
from platform_proofs.scenarios.verified_product_identification.ingest.source_reader.parquet_dataset import (
    iter_dataset_rows,
)


def sample_semantic_texts(
    config: VpiEmbeddingMaterializationConfig,
    *,
    record_count: int,
) -> tuple[str, ...]:
    if record_count <= 0:
        msg = "record_count must be > 0"
        raise ValueError(msg)
    texts: list[str] = []
    for _, rows in iter_dataset_rows(
        config.dataset_path,
        batch_size=config.source_read_batch_size,
        start_row_index=0,
        start_batch_ordinal=0,
        max_records=record_count,
    ):
        catalog_batch = build_catalog_ingest_batch(
            batch_ordinal=0,
            rows=rows,
            catalog_id=config.catalog_id,
            source_revision=config.source_revision,
        )
        for record in catalog_batch.records:
            texts.append(record.representation.semantic.semantic_text)
            if len(texts) >= record_count:
                return tuple(texts)
    if len(texts) < record_count:
        msg = (
            f"dataset yielded only {len(texts)} semantic texts; "
            f"requested {record_count}"
        )
        raise RuntimeError(msg)
    return tuple(texts)
