"""Build proof-50 universal data pack."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    load_vpi_embedding_configuration,
    validate_resolved_provider_dimension,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    assert_execution_device_available,
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
    write_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.sample_selection import (
    PROOF_50_SAMPLE_SEED,
    SelectedDatasetRow,
    select_proof_sample_rows,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.validation import (
    assert_validation_pass,
    validate_cross_artifact_identity,
    validate_embedding_records,
    validate_relational_records,
    validate_semantic_text_hashes,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_mode import (
    DataPackBuildMode,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.content_identity import (
    compute_data_pack_content_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    EmbeddingModelIdentityError,
    VpiDataPackBuildError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    PARQUET_FILE_FORMAT,
    PROOF_50_RECORD_COUNT,
    PROOF_50_SAMPLE_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    SCENARIO_ID,
    semantic_text_hash,
    source_ref_key,
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    BuildExecutionProvenance,
    DataPackManifest,
    EmbeddingPackIdentity,
    SampleIdentity,
    SourceDatasetIdentity,
    write_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DataPackPaths,
    final_shard_path,
    resolve_data_pack_paths,
    shard_file_name,
    temp_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
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
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.model_identity import (
    resolve_embedding_model_identity,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)


def _load_dataset_identity(dataset_manifest_path: Path) -> SourceDatasetIdentity:
    payload = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VpiDataPackBuildError("dataset manifest must be a JSON object")
    return SourceDatasetIdentity(
        dataset_name=str(payload.get("source_dataset_name", "offers_corpus_all_v2_non_norm")),
        dataset_path=str(payload.get("output_path", "")),
        dataset_sha256=str(payload.get("output_sha256", "")),
        dataset_record_count=int(payload.get("selected_record_count", 0)),
    )


def _resolve_model_revision(
    *,
    provider: str,
    model: str,
    build_mode: DataPackBuildMode,
) -> tuple[str, str | None]:
    try:
        resolved = resolve_embedding_model_identity(provider, model)
    except EmbeddingModelIdentityError as exc:
        if build_mode is DataPackBuildMode.CANONICAL:
            raise VpiDataPackBuildError(str(exc)) from exc
        raise
    if build_mode is DataPackBuildMode.CANONICAL and not resolved.revision.strip():
        raise VpiDataPackBuildError("canonical data pack requires non-null embedding model revision")
    return resolved.revision, resolved.artifact_fingerprint


def _derive_relational_record(
    row: SelectedDatasetRow,
    *,
    catalog_id: str,
    source_revision: str | None,
) -> RelationalDataPackRecord:
    source_offer = parse_wdc_source_offer_json(row.record_json)
    source_ref = build_source_record_ref(
        source_offer,
        catalog_id=catalog_id,
        source_revision=source_revision,
    )
    representation = derive_search_representation(source_offer, source_ref=source_ref)
    semantic_text = representation.semantic.semantic_text
    return RelationalDataPackRecord(
        global_row_index=row.global_row_index,
        source_ref=source_ref,
        record_json=row.record_json,
        derivation_version=representation.derivation_version,
        semantic_text=semantic_text,
        semantic_text_hash=semantic_text_hash(semantic_text),
        title=source_offer.title,
        brand=source_offer.brand,
        category=source_offer.category,
        description=source_offer.description,
        has_identifiers=len(source_offer.identifiers) > 0,
        has_spec_table=source_offer.spec_table_content is not None
        and bool(source_offer.spec_table_content.strip()),
        has_structured_attributes=len(source_offer.key_value_pairs) > 0,
    )


def _sort_relational_records(
    records: Sequence[RelationalDataPackRecord],
) -> tuple[RelationalDataPackRecord, ...]:
    return tuple(sorted(records, key=lambda record: record.global_row_index))


def _write_shard_atomically(
    directory: Path,
    shard_ordinal: int,
    write_callable,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    temp_path = temp_shard_path(directory, shard_ordinal)
    final_path = final_shard_path(directory, shard_ordinal)
    if temp_path.exists():
        temp_path.unlink()
    write_callable(temp_path)
    temp_path.replace(final_path)
    return final_path


def build_proof_50_data_pack(
    *,
    output_root: Path,
    dataset_path: Path,
    dataset_manifest_path: Path,
    catalog_id: str = "wdc-v2-selected",
    source_revision: str | None = None,
    record_count: int = PROOF_50_RECORD_COUNT,
    build_mode: DataPackBuildMode = DataPackBuildMode.CANONICAL,
) -> DataPackManifest:
    paths = resolve_data_pack_paths(output_root)
    for directory in (
        paths.manifest_dir,
        paths.relational_dir,
        paths.embeddings_dir,
        paths.indexes_dir,
        paths.checksums_dir,
        paths.evidence_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_identity = _load_dataset_identity(dataset_manifest_path)
    selected_rows = select_proof_sample_rows(str(dataset_path), record_count=record_count)
    relational_records = _sort_relational_records(
        tuple(
            _derive_relational_record(
                row,
                catalog_id=catalog_id,
                source_revision=source_revision,
            )
            for row in selected_rows
        )
    )

    ensure_embedding_provider_integrations_registered()
    embedding_configuration = load_vpi_embedding_configuration()
    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    assert_execution_device_available(execution_configuration)
    embedding_adapter = IntergraxEmbeddingBootstrapAdapter(
        embedding_configuration,
        execution_configuration=execution_configuration,
    )
    model = embedding_configuration.model
    if model is None:
        raise VpiDataPackBuildError("embedding model is required")
    probe = embedding_adapter.probe()
    validate_resolved_provider_dimension(
        configuration=embedding_configuration,
        resolved_dimension=probe.resolved_dimension,
    )
    model_revision, artifact_fingerprint = _resolve_model_revision(
        provider=embedding_configuration.provider,
        model=model,
        build_mode=build_mode,
    )

    semantic_texts = [record.semantic_text for record in relational_records]
    vectors = embedding_adapter.embed_batch(semantic_texts)
    if len(vectors) != len(relational_records):
        raise VpiDataPackBuildError("embedding batch size mismatch")

    embedding_records: list[EmbeddingDataPackRecord] = []
    for relational_record, vector in zip(relational_records, vectors, strict=True):
        source_ref = relational_record.source_ref
        embedding_records.append(
            EmbeddingDataPackRecord(
                logical_point_id=search_representation_point_id(
                    catalog_id=source_ref.catalog_id,
                    offer_id=source_ref.offer_id.value,
                    derivation_version=relational_record.derivation_version,
                ),
                source_ref=source_ref,
                derivation_version=relational_record.derivation_version,
                semantic_text_hash=relational_record.semantic_text_hash,
                embedding_provider=embedding_configuration.provider,
                embedding_model=model,
                embedding_model_revision=model_revision,
                embedding_dimension=embedding_configuration.expected_dimension,
                dense_embedding=vector,
            )
        )
    embedding_adapter.close()
    embedding_tuple = tuple(embedding_records)

    expected_refs = frozenset(source_ref_key(record.source_ref) for record in relational_records)
    assert_validation_pass(
        validate_relational_records(relational_records, expected_count=record_count),
        stage="relational_validation",
    )
    assert_validation_pass(
        validate_embedding_records(
            embedding_tuple,
            expected_count=record_count,
            expected_dimension=embedding_configuration.expected_dimension,
        ),
        stage="embedding_validation",
    )
    assert_validation_pass(
        validate_cross_artifact_identity(
            relational_records,
            embedding_tuple,
            expected_refs=expected_refs,
        ),
        stage="cross_ref_validation",
    )
    assert_validation_pass(
        validate_semantic_text_hashes(relational_records, embedding_tuple),
        stage="semantic_text_hash_validation",
    )

    shard_ordinal = 1
    relational_file = shard_file_name(shard_ordinal)
    embedding_file = shard_file_name(shard_ordinal)
    relational_path = _write_shard_atomically(
        paths.relational_dir,
        shard_ordinal,
        lambda temp_path: write_relational_parquet(temp_path, relational_records),
    )
    embedding_path = _write_shard_atomically(
        paths.embeddings_dir,
        shard_ordinal,
        lambda temp_path: write_embedding_parquet(
            temp_path,
            embedding_tuple,
            embedding_dimension=embedding_configuration.expected_dimension,
        ),
    )

    selected_ref_labels = tuple(
        f"{record.source_ref.catalog_id}:{record.source_ref.offer_id.value}"
        for record in relational_records
    )
    created_at = datetime.now(UTC).isoformat()
    embedding_identity = EmbeddingPackIdentity(
        provider=embedding_configuration.provider,
        model=model,
        model_revision=model_revision,
        artifact_fingerprint=artifact_fingerprint,
        dimension=embedding_configuration.expected_dimension,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        input_policy_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
    )
    content_identity = compute_data_pack_content_identity(
        source_dataset=dataset_identity,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    manifest_base = dict(
        data_pack_version=DATA_PACK_VERSION,
        content_identity=content_identity,
        scenario_id=SCENARIO_ID,
        source_dataset=dataset_identity,
        source_record_count=record_count,
        sample_identity=SampleIdentity(
            sample_version=PROOF_50_SAMPLE_VERSION,
            sample_seed=PROOF_50_SAMPLE_SEED,
            selected_record_refs=selected_ref_labels,
        ),
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        relational_format=PARQUET_FILE_FORMAT,
        embedding_format=PARQUET_FILE_FORMAT,
        shard_count=1,
        record_count=record_count,
        created_at_utc=created_at,
        checksums_path="checksums/SHA256SUMS",
        shards_index_path="indexes/shards.json",
        build_execution_provenance=BuildExecutionProvenance(
            device=execution_configuration.device,
            provider_batch_size=execution_configuration.provider_batch_size,
        ),
    )

    relational_descriptor = ShardDescriptor(
        ordinal=shard_ordinal,
        relative_path=f"relational/{relational_file}",
        record_count=record_count,
        sha256=sha256_file(relational_path),
        source_ref_count=record_count,
        source_ref_set_sha256=source_ref_set_sha256(
            tuple(record.source_ref for record in relational_records)
        ),
        schema_version=RELATIONAL_SCHEMA_VERSION,
    )
    embedding_descriptor = ShardDescriptor(
        ordinal=shard_ordinal,
        relative_path=f"embeddings/{embedding_file}",
        record_count=record_count,
        sha256=sha256_file(embedding_path),
        source_ref_count=record_count,
        source_ref_set_sha256=source_ref_set_sha256(
            tuple(record.source_ref for record in embedding_tuple)
        ),
        schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    shard_index = ShardIndex(
        shard_count=1,
        relational_shards=(relational_descriptor,),
        embedding_shards=(embedding_descriptor,),
    )
    write_shard_index_file(paths.shards_index_file, shard_index)

    ready_manifest = DataPackManifest(status=DataPackStatus.READY, **manifest_base)
    write_manifest_file(paths.manifest_file, ready_manifest)

    write_sha256sums(
        paths.checksums_file,
        (
            ("manifest/manifest.json", paths.manifest_file),
            (relational_descriptor.relative_path, relational_path),
            (embedding_descriptor.relative_path, embedding_path),
            ("indexes/shards.json", paths.shards_index_file),
        ),
    )
    return ready_manifest
