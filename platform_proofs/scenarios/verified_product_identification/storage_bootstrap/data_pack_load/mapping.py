"""Map canonical Data Pack records into provider-neutral load records."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalLoadRecord,
    VectorLoadRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIdentityError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.ports import (
    PairedDataPackRecord,
)


def identity_key(source_ref: SourceRecordRef) -> str:
    catalog_id, offer_id, source_revision = source_ref_key(source_ref)
    revision = source_revision or ""
    return f"{catalog_id}:{offer_id}:{revision}"


def relational_load_record_from_pack(record: RelationalDataPackRecord) -> RelationalLoadRecord:
    return RelationalLoadRecord(
        source_ref=record.source_ref,
        global_row_index=record.global_row_index,
        record_json=record.record_json,
        semantic_text=record.semantic_text,
        semantic_text_hash=record.semantic_text_hash,
        derivation_version=record.derivation_version,
    )


def vector_load_record_from_pack(record: EmbeddingDataPackRecord) -> VectorLoadRecord:
    return VectorLoadRecord(
        logical_point_id=record.logical_point_id,
        source_ref=record.source_ref,
        semantic_text_hash=record.semantic_text_hash,
        embedding_provider=record.embedding_provider,
        embedding_model=record.embedding_model,
        embedding_revision=record.embedding_model_revision,
        embedding_dimension=record.embedding_dimension,
        dense_embedding=record.dense_embedding,
        derivation_version=record.derivation_version,
    )


def assert_paired_identity(pair: PairedDataPackRecord) -> None:
    relational = pair.relational
    embedding = pair.embedding
    if relational.source_ref != embedding.source_ref:
        raise StorageBootstrapIdentityError(
            f"source_ref mismatch at row {relational.global_row_index}: "
            f"relational={identity_key(relational.source_ref)} "
            f"embedding={identity_key(embedding.source_ref)}"
        )
    if relational.semantic_text_hash != embedding.semantic_text_hash:
        raise StorageBootstrapIdentityError(
            f"semantic_text_hash mismatch at row {relational.global_row_index}"
        )
    if relational.derivation_version != embedding.derivation_version:
        raise StorageBootstrapIdentityError(
            f"derivation_version mismatch at row {relational.global_row_index}"
        )


def paired_load_records(
    pair: PairedDataPackRecord,
) -> tuple[RelationalLoadRecord, VectorLoadRecord]:
    assert_paired_identity(pair)
    return (
        relational_load_record_from_pack(pair.relational),
        vector_load_record_from_pack(pair.embedding),
    )
