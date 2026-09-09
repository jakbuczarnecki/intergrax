"""Shared fixtures for VPI storage bootstrap runtime qualification."""

from __future__ import annotations

import uuid
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
    write_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.content_identity import (
    compute_data_pack_content_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    EmbeddingPackIdentity,
    SourceDatasetIdentity,
    write_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    final_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardDescriptor,
    ShardIndex,
    write_shard_index_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    write_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    write_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.configuration import (
    CANONICAL_EMBEDDING_DIMENSION,
    CANONICAL_EMBEDDING_MODEL,
    CANONICAL_EMBEDDING_PROVIDER,
    CANONICAL_EMBEDDING_REVISION,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.adapter import (
    PgVectorStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_storage_bootstrap_data_pack_load import (
    _build_relational,
)

_RUNTIME_PREFIX = "vpi_5c5e_runtime"


def runtime_qualification_target_name() -> str:
    return f"{_RUNTIME_PREFIX}_{uuid.uuid4().hex[:8]}"


def deterministic_dense_embedding(
    global_row_index: int,
    *,
    dimension: int = CANONICAL_EMBEDDING_DIMENSION,
) -> tuple[float, ...]:
    values = [0.0] * dimension
    slot = global_row_index % dimension
    values[slot] = float((global_row_index % 997) + 1) * 0.001
    return tuple(values)


def _build_embedding_record(
    relational: RelationalDataPackRecord,
    *,
    dimension: int = CANONICAL_EMBEDDING_DIMENSION,
) -> EmbeddingDataPackRecord:
    return EmbeddingDataPackRecord(
        logical_point_id=search_representation_point_id(
            catalog_id=relational.source_ref.catalog_id,
            offer_id=relational.source_ref.offer_id.value,
            derivation_version=relational.derivation_version,
        ),
        source_ref=relational.source_ref,
        derivation_version=relational.derivation_version,
        semantic_text_hash=relational.semantic_text_hash,
        embedding_provider=CANONICAL_EMBEDDING_PROVIDER,
        embedding_model=CANONICAL_EMBEDDING_MODEL,
        embedding_model_revision=CANONICAL_EMBEDDING_REVISION,
        embedding_dimension=dimension,
        dense_embedding=deterministic_dense_embedding(
            relational.global_row_index,
            dimension=dimension,
        ),
    )


def _runtime_ready_manifest(record_count: int) -> DataPackManifest:
    source_dataset = SourceDatasetIdentity(
        dataset_name="offers",
        dataset_path="/tmp/runtime-qualification.parquet",
        dataset_sha256="runtime-qualification-sha256",
        dataset_record_count=record_count,
    )
    embedding_identity = EmbeddingPackIdentity(
        provider=CANONICAL_EMBEDDING_PROVIDER,
        model=CANONICAL_EMBEDDING_MODEL,
        model_revision=CANONICAL_EMBEDDING_REVISION,
        artifact_fingerprint=None,
        dimension=CANONICAL_EMBEDDING_DIMENSION,
        embedding_configuration_version="v1",
        input_policy_version="v2",
    )
    content_identity = compute_data_pack_content_identity(
        source_dataset=source_dataset,
        derivation_version="v2",
        semantic_text_version="v2",
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    return DataPackManifest(
        data_pack_version="vpi.data_pack/1.0.0",
        content_identity=content_identity,
        scenario_id="verified_product_identification",
        source_dataset=source_dataset,
        source_record_count=record_count,
        sample_identity=None,
        derivation_version="v2",
        semantic_text_version="v2",
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        relational_format="parquet",
        embedding_format="parquet",
        shard_count=1,
        record_count=record_count,
        created_at_utc="2026-09-09T00:00:00+00:00",
        status=DataPackStatus.READY,
        checksums_path="checksums/SHA256SUMS",
        shards_index_path="indexes/shards.json",
        build_execution_provenance=None,
    )


def write_runtime_qualification_pack(
    pack_root: Path,
    *,
    record_count: int,
    shard_size: int = 5,
) -> Path:
    relational_shards: list[ShardDescriptor] = []
    embedding_shards: list[ShardDescriptor] = []
    shard_count = max(1, (record_count + shard_size - 1) // shard_size) if record_count else 1
    for ordinal in range(1, shard_count + 1):
        start = (ordinal - 1) * shard_size
        end = min(start + shard_size, record_count)
        relational_records_list: list[RelationalDataPackRecord] = []
        embedding_records_list: list[EmbeddingDataPackRecord] = []
        for index in range(start, end):
            relational = _build_relational(index, str(index))
            relational_records_list.append(relational)
            embedding_records_list.append(_build_embedding_record(relational))
        if not relational_records_list:
            continue
        relational_path = final_shard_path(pack_root / "relational", ordinal)
        embedding_path = final_shard_path(pack_root / "embeddings", ordinal)
        relational_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        typed_relational = tuple(relational_records_list)
        embedding_records = tuple(embedding_records_list)
        write_relational_parquet(relational_path, typed_relational)
        write_embedding_parquet(
            embedding_path,
            embedding_records,
            embedding_dimension=CANONICAL_EMBEDDING_DIMENSION,
        )
        digest = source_ref_set_sha256(tuple(record.source_ref for record in typed_relational))
        relational_shards.append(
            ShardDescriptor(
                ordinal=ordinal,
                relative_path=f"relational/part-{ordinal:06d}.parquet",
                record_count=len(typed_relational),
                sha256=sha256_file(relational_path),
                source_ref_count=len(typed_relational),
                source_ref_set_sha256=digest,
                schema_version=RELATIONAL_SCHEMA_VERSION,
            )
        )
        embedding_shards.append(
            ShardDescriptor(
                ordinal=ordinal,
                relative_path=f"embeddings/part-{ordinal:06d}.parquet",
                record_count=len(embedding_records),
                sha256=sha256_file(embedding_path),
                source_ref_count=len(embedding_records),
                source_ref_set_sha256=digest,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            )
        )
    manifest = _runtime_ready_manifest(record_count)
    manifest_path = pack_root / "manifest" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_file(manifest_path, manifest)
    shard_index_path = pack_root / "indexes" / "shards.json"
    write_shard_index_file(
        shard_index_path,
        ShardIndex(
            shard_count=len(relational_shards),
            relational_shards=tuple(relational_shards),
            embedding_shards=tuple(embedding_shards),
        ),
    )
    checksum_entries: list[tuple[str, Path]] = [
        ("manifest/manifest.json", manifest_path),
        ("indexes/shards.json", shard_index_path),
    ]
    for descriptor in relational_shards:
        checksum_entries.append(
            (descriptor.relative_path, pack_root / descriptor.relative_path)
        )
    for descriptor in embedding_shards:
        checksum_entries.append(
            (descriptor.relative_path, pack_root / descriptor.relative_path)
        )
    checksums_dir = pack_root / "checksums"
    checksums_dir.mkdir(parents=True, exist_ok=True)
    write_sha256sums(checksums_dir / "SHA256SUMS", tuple(checksum_entries))
    return pack_root


def drop_postgresql_schema(adapter: PostgreSqlRelationalStorageAdapter) -> None:
    schema_name = adapter._configuration.schema_name
    with adapter._provider.connection() as session:
        session.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')


def drop_pgvector_schema(adapter: PgVectorStorageAdapter) -> None:
    schema_name = adapter._configuration.schema_name
    with adapter._provider.connection() as session:
        session.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
