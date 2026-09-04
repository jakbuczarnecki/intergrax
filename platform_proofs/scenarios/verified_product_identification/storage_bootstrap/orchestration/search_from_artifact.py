"""Build search ingest records directly from materialized artifact rows."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    SearchIndexIngestRecord,
)


def search_ingest_record_from_artifact(
    artifact_record: EmbeddingArtifactRecord,
    *,
    dataset_checksum: str,
) -> SearchIndexIngestRecord:
    return SearchIndexIngestRecord(
        logical_point_id=artifact_record.logical_point_id,
        dense_embedding=artifact_record.dense_embedding,
        lexical_text=artifact_record.lexical_text,
        source_ref=artifact_record.source_ref(),
        derivation_version=artifact_record.derivation_version,
        dataset_checksum=dataset_checksum,
        embedding_provider=artifact_record.embedding_provider,
        embedding_model=artifact_record.embedding_model,
        embedding_dimension=artifact_record.embedding_dimension,
    )
