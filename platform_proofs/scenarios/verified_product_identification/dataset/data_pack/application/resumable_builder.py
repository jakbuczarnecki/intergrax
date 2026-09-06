"""Resumable multi-shard VPI Data Pack v1 builder."""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

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
from platform_proofs.scenarios.verified_product_identification.data_package.identity import (
    CANONICAL_CATALOG_ID,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_progress import (
    DataPackBuildProgress,
    compute_build_progress,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_state_machine import (
    discard_shard_temp_outputs,
    persist_build_state,
    recover_non_ready_shard,
    replace_shard,
    shard_descriptor_paths,
    transition_shard,
    validate_ready_shard_artifacts,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
    write_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.compatibility import (
    assert_data_pack_compatible,
    default_v1_expectations,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.sample_selection import (
    SelectedDatasetRow,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_plan import (
    plan_data_pack_shards,
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
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackBuildState,
    DataPackShardBuildState,
    DataPackShardStatus,
    VPI_DATA_PACK_BUILD_STATE_VERSION,
    read_build_state_file,
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
    VpiDataPackBuildIdentityMismatchError,
    VpiDataPackBuildStateError,
    VpiDataPackResumeError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    PARQUET_FILE_FORMAT,
    RELATIONAL_SCHEMA_VERSION,
    SCENARIO_ID,
    VPI_CANONICAL_EMBEDDING_DIMENSION,
    VPI_CANONICAL_EMBEDDING_MODEL,
    VPI_CANONICAL_EMBEDDING_PROVIDER,
    VPI_CANONICAL_EMBEDDING_REVISION,
    semantic_text_hash,
    source_ref_key,
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    BuildExecutionProvenance,
    DataPackManifest,
    EmbeddingPackIdentity,
    SourceDatasetIdentity,
    write_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DataPackPaths,
    final_shard_path,
    resolve_data_pack_paths,
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
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.selected_dataset_reader import (
    SelectedDatasetShardReaderPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    write_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    write_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.orchestration.embedding_batches import (
    iter_embedding_slices,
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

logger = logging.getLogger(__name__)

_BUILD_SUBDIRS = (
    "manifest",
    "relational",
    "embeddings",
    "indexes",
    "checksums",
    "evidence",
    "state",
)


class DataPackEmbeddingPort(Protocol):
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DataPackBuildConfig:
    output_root: Path
    dataset_path: Path
    dataset_manifest_path: Path
    shard_size: int
    catalog_id: str = CANONICAL_CATALOG_ID
    source_revision: str | None = None
    max_records: int | None = None
    max_shards: int | None = None
    resume: bool = False
    start_fresh: bool = False
    stop_after_shard: int | None = None
    build_mode: DataPackBuildMode = DataPackBuildMode.CANONICAL
    reader_batch_size: int = 4096


@dataclass(frozen=True, slots=True)
class DataPackBuildReport:
    status: DataPackStatus
    manifest: DataPackManifest | None
    progress: DataPackBuildProgress
    records_per_second: float | None
    embedding_records_per_second: float | None
    average_relational_shard_bytes: int | None
    average_embedding_shard_bytes: int | None
    finalized: bool


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
    if build_mode is DataPackBuildMode.CANONICAL and resolved.revision != VPI_CANONICAL_EMBEDDING_REVISION:
        raise VpiDataPackBuildError(
            "canonical embedding revision mismatch: "
            f"expected {VPI_CANONICAL_EMBEDDING_REVISION}, resolved {resolved.revision}"
        )
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


def _ensure_build_directories(paths: DataPackPaths) -> None:
    for name in _BUILD_SUBDIRS:
        (paths.root / name).mkdir(parents=True, exist_ok=True)


def _output_root_has_unexpected_content(root: Path) -> bool:
    if not root.exists():
        return False
    for child in root.iterdir():
        if child.name not in _BUILD_SUBDIRS:
            return True
        if child.is_file():
            return True
    return False


def _clear_build_contents(root: Path) -> None:
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return
    for name in _BUILD_SUBDIRS:
        target = root / name
        if target.exists():
            shutil.rmtree(target)
    for child in root.iterdir():
        if child.is_file():
            child.unlink()


def _new_build_state(
    *,
    content_identity: str,
    expected_record_count: int,
    shard_size: int,
    catalog_id: str,
) -> DataPackBuildState:
    now = datetime.now(UTC).isoformat()
    plan = plan_data_pack_shards(record_count=expected_record_count, shard_size=shard_size)
    shards = tuple(
        DataPackShardBuildState(
            ordinal=entry.ordinal,
            start_row_index=entry.start_row_index,
            end_row_index_exclusive=entry.end_row_index_exclusive,
            expected_record_count=entry.expected_record_count,
            status=DataPackShardStatus.PENDING,
            relational_relative_path=None,
            embedding_relative_path=None,
            attempt=0,
        )
        for entry in plan
    )
    return DataPackBuildState(
        state_version=VPI_DATA_PACK_BUILD_STATE_VERSION,
        build_id=f"vpi-data-pack-{uuid.uuid4()}",
        content_identity=content_identity,
        expected_record_count=expected_record_count,
        shard_size=shard_size,
        shard_count=len(shards),
        catalog_id=catalog_id,
        started_at_utc=now,
        updated_at_utc=now,
        completed_shards=0,
        shards=shards,
    )


def _persist_state(paths: DataPackPaths, state: DataPackBuildState) -> DataPackBuildState:
    updated = replace(state, updated_at_utc=datetime.now(UTC).isoformat())
    persist_build_state(paths.build_state_file, updated)
    return updated


def _record_failure(
    state: DataPackBuildState,
    shard: DataPackShardBuildState,
    *,
    error_code: str,
    message: str,
) -> DataPackBuildState:
    failed_shard = replace(
        shard,
        last_error_code=error_code,
        last_error_message=message[:500],
    )
    return replace_shard(state, failed_shard)


def estimate_required_disk_bytes(
    *,
    record_count: int,
    shard_count: int,
    average_relational_shard_bytes: int | None,
    average_embedding_shard_bytes: int | None,
) -> int | None:
    if average_relational_shard_bytes is None or average_embedding_shard_bytes is None:
        return None
    overhead = 64 * 1024
    return (
        shard_count * (average_relational_shard_bytes + average_embedding_shard_bytes)
        + overhead
    )


def assert_sufficient_disk_space(path: Path, required_bytes: int | None) -> None:
    if required_bytes is None:
        return
    usage = shutil.disk_usage(path)
    if usage.free < required_bytes:
        raise VpiDataPackBuildError(
            f"insufficient disk space at {path}: required ~{required_bytes} bytes, "
            f"available {usage.free} bytes"
        )


def _create_default_embedding_port() -> DataPackEmbeddingPort:
    ensure_embedding_provider_integrations_registered()
    embedding_configuration = load_vpi_embedding_configuration()
    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    assert_execution_device_available(execution_configuration)
    adapter = IntergraxEmbeddingBootstrapAdapter(
        embedding_configuration,
        execution_configuration=execution_configuration,
    )
    probe = adapter.probe()
    validate_resolved_provider_dimension(
        configuration=embedding_configuration,
        resolved_dimension=probe.resolved_dimension,
    )
    return adapter


def _embed_shard_texts(
    embedding_port: DataPackEmbeddingPort,
    *,
    semantic_texts: Sequence[str],
    provider_batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for _start, batch in iter_embedding_slices(semantic_texts, batch_size=provider_batch_size):
        vectors.extend(embedding_port.embed_batch(batch))
    if len(vectors) != len(semantic_texts):
        raise VpiDataPackBuildError("embedding batch size mismatch")
    return vectors


def _build_embedding_records(
    relational_records: Sequence[RelationalDataPackRecord],
    vectors: Sequence[Sequence[float]],
    *,
    provider: str,
    model: str,
    model_revision: str,
    dimension: int,
) -> tuple[EmbeddingDataPackRecord, ...]:
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
                embedding_provider=provider,
                embedding_model=model,
                embedding_model_revision=model_revision,
                embedding_dimension=dimension,
                dense_embedding=list(vector),
            )
        )
    return tuple(embedding_records)


def _finalize_data_pack(
    *,
    paths: DataPackPaths,
    state: DataPackBuildState,
    dataset_identity: SourceDatasetIdentity,
    embedding_identity: EmbeddingPackIdentity,
    execution_provenance: BuildExecutionProvenance,
) -> DataPackManifest:
    relational_descriptors: list[ShardDescriptor] = []
    embedding_descriptors: list[ShardDescriptor] = []
    for shard in state.shards:
        if shard.relational_relative_path is None or shard.embedding_relative_path is None:
            raise VpiDataPackBuildError(f"READY shard {shard.ordinal} missing descriptor paths")
        if shard.relational_sha256 is None or shard.embedding_sha256 is None:
            raise VpiDataPackBuildError(f"READY shard {shard.ordinal} missing sha256 metadata")
        if (
            shard.relational_source_ref_set_sha256 is None
            or shard.embedding_source_ref_set_sha256 is None
        ):
            raise VpiDataPackBuildError(f"READY shard {shard.ordinal} missing source-ref digests")
        relational_descriptors.append(
            ShardDescriptor(
                ordinal=shard.ordinal,
                relative_path=shard.relational_relative_path,
                record_count=shard.expected_record_count,
                sha256=shard.relational_sha256,
                source_ref_count=shard.expected_record_count,
                source_ref_set_sha256=shard.relational_source_ref_set_sha256,
                schema_version=RELATIONAL_SCHEMA_VERSION,
            )
        )
        embedding_descriptors.append(
            ShardDescriptor(
                ordinal=shard.ordinal,
                relative_path=shard.embedding_relative_path,
                record_count=shard.expected_record_count,
                sha256=shard.embedding_sha256,
                source_ref_count=shard.expected_record_count,
                source_ref_set_sha256=shard.embedding_source_ref_set_sha256,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            )
        )

    shard_index = ShardIndex(
        shard_count=state.shard_count,
        relational_shards=tuple(relational_descriptors),
        embedding_shards=tuple(embedding_descriptors),
    )
    write_shard_index_file(paths.shards_index_file, shard_index)

    content_identity = compute_data_pack_content_identity(
        source_dataset=dataset_identity,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    created_at = datetime.now(UTC).isoformat()
    manifest_base = dict(
        data_pack_version=DATA_PACK_VERSION,
        content_identity=content_identity,
        scenario_id=SCENARIO_ID,
        source_dataset=dataset_identity,
        source_record_count=state.expected_record_count,
        sample_identity=None,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        relational_format=PARQUET_FILE_FORMAT,
        embedding_format=PARQUET_FILE_FORMAT,
        shard_count=state.shard_count,
        record_count=state.expected_record_count,
        created_at_utc=created_at,
        checksums_path="checksums/SHA256SUMS",
        shards_index_path="indexes/shards.json",
        build_execution_provenance=execution_provenance,
    )
    ready_manifest = DataPackManifest(status=DataPackStatus.READY, **manifest_base)
    write_manifest_file(paths.manifest_file, ready_manifest)
    checksum_entries: list[tuple[str, Path]] = [
        ("manifest/manifest.json", paths.manifest_file),
        ("indexes/shards.json", paths.shards_index_file),
    ]
    for descriptor in relational_descriptors:
        checksum_entries.append((descriptor.relative_path, paths.root / descriptor.relative_path))
    for descriptor in embedding_descriptors:
        checksum_entries.append((descriptor.relative_path, paths.root / descriptor.relative_path))
    write_sha256sums(paths.checksums_file, tuple(checksum_entries))

    assert_data_pack_compatible(
        ready_manifest,
        expectations=default_v1_expectations(
            derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
            semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
            embedding_provider=embedding_identity.provider,
            embedding_model=embedding_identity.model,
            embedding_model_revision=embedding_identity.model_revision or "",
            embedding_dimension=embedding_identity.dimension,
            source_dataset_sha256=dataset_identity.dataset_sha256,
        ),
        pack_root=paths.root,
    )
    return ready_manifest


def run_resumable_data_pack_build(
    config: DataPackBuildConfig,
    *,
    embedding_port: DataPackEmbeddingPort | None = None,
) -> DataPackBuildReport:
    paths = resolve_data_pack_paths(config.output_root)
    dataset_identity = _load_dataset_identity(config.dataset_manifest_path)
    expected_record_count = dataset_identity.dataset_record_count
    if config.max_records is not None:
        expected_record_count = min(expected_record_count, config.max_records)

    embedding_configuration = load_vpi_embedding_configuration()
    model = embedding_configuration.model
    if model is None:
        raise VpiDataPackBuildError("embedding model is required")
    if embedding_configuration.provider != VPI_CANONICAL_EMBEDDING_PROVIDER:
        raise VpiDataPackBuildError("canonical build requires hf embedding provider")
    if model != VPI_CANONICAL_EMBEDDING_MODEL:
        raise VpiDataPackBuildError("canonical build requires BAAI/bge-m3 model")
    if embedding_configuration.expected_dimension != VPI_CANONICAL_EMBEDDING_DIMENSION:
        raise VpiDataPackBuildError("canonical build requires embedding dimension 1024")

    model_revision, artifact_fingerprint = _resolve_model_revision(
        provider=embedding_configuration.provider,
        model=model,
        build_mode=config.build_mode,
    )
    embedding_identity = EmbeddingPackIdentity(
        provider=embedding_configuration.provider,
        model=model,
        model_revision=model_revision,
        artifact_fingerprint=artifact_fingerprint,
        dimension=embedding_configuration.expected_dimension,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        input_policy_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
    )
    expected_content_identity = compute_data_pack_content_identity(
        source_dataset=dataset_identity,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )

    if config.start_fresh:
        _clear_build_contents(paths.root)

    state_exists = paths.build_state_file.is_file()
    if state_exists and not config.resume and not config.start_fresh:
        raise VpiDataPackResumeError(
            "existing build state found; pass --resume or --start-fresh explicitly"
        )
    if not state_exists and _output_root_has_unexpected_content(paths.root):
        raise VpiDataPackResumeError(
            "output directory is non-empty without build state authority"
        )

    _ensure_build_directories(paths)

    if state_exists:
        state = read_build_state_file(paths.build_state_file)
        if state.content_identity != expected_content_identity:
            raise VpiDataPackBuildIdentityMismatchError(
                "build state content_identity does not match expected dataset/model identity"
            )
        if state.shard_size != config.shard_size:
            raise VpiDataPackResumeError("build state shard_size does not match requested shard_size")
        if state.expected_record_count != expected_record_count:
            raise VpiDataPackResumeError(
                "build state expected_record_count does not match current configuration"
            )
        recovered_shards = tuple(
            recover_non_ready_shard(
                shard,
                relational_dir=paths.relational_dir,
                embeddings_dir=paths.embeddings_dir,
            )
            for shard in state.shards
        )
        state = replace(state, shards=recovered_shards)
        state = _persist_state(paths, state)
    else:
        state = _new_build_state(
            content_identity=expected_content_identity,
            expected_record_count=expected_record_count,
            shard_size=config.shard_size,
            catalog_id=config.catalog_id,
        )
        state = _persist_state(paths, state)

    reader = SelectedDatasetShardReaderPort(
        config.dataset_path,
        batch_size=config.reader_batch_size,
    )
    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    provider_batch_size = execution_configuration.provider_batch_size or 16
    execution_provenance = BuildExecutionProvenance(
        device=execution_configuration.device,
        provider_batch_size=provider_batch_size,
    )

    owns_embedding_port = embedding_port is None
    active_embedding_port = embedding_port or _create_default_embedding_port()

    started = time.perf_counter()
    embedding_started = time.perf_counter()
    records_embedded = 0
    relational_bytes: list[int] = []
    embedding_bytes: list[int] = []
    manifest: DataPackManifest | None = None
    finalized = False

    shard_limit = config.max_shards
    if config.stop_after_shard is not None:
        shard_limit = config.stop_after_shard

    try:
        for shard in state.shards:
            if shard_limit is not None and shard.ordinal > shard_limit:
                break

            if shard.status is DataPackShardStatus.READY:
                validate_ready_shard_artifacts(
                    pack_root=paths.root,
                    shard=shard,
                    relational_schema_version=RELATIONAL_SCHEMA_VERSION,
                    embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
                    expected_dimension=embedding_configuration.expected_dimension,
                )
                logger.info("skip READY shard ordinal=%s records=%s", shard.ordinal, shard.expected_record_count)
                continue

            current = shard
            try:
                current = transition_shard(current, DataPackShardStatus.DERIVING)
                state = replace_shard(state, current)
                state = _persist_state(paths, state)

                selected_rows = tuple(
                    reader.read_range(current.start_row_index, current.end_row_index_exclusive)
                )
                relational_records = _sort_relational_records(
                    tuple(
                        _derive_relational_record(
                            row,
                            catalog_id=config.catalog_id,
                            source_revision=config.source_revision,
                        )
                        for row in selected_rows
                    )
                )
                expected_refs = frozenset(source_ref_key(record.source_ref) for record in relational_records)
                assert_validation_pass(
                    validate_relational_records(
                        relational_records,
                        expected_count=current.expected_record_count,
                    ),
                    stage="relational_validation",
                )

                current = transition_shard(current, DataPackShardStatus.EMBEDDING)
                state = replace_shard(state, current)
                state = _persist_state(paths, state)

                semantic_texts = [record.semantic_text for record in relational_records]
                vectors = _embed_shard_texts(
                    active_embedding_port,
                    semantic_texts=semantic_texts,
                    provider_batch_size=provider_batch_size,
                )
                records_embedded += len(vectors)
                embedding_records = _build_embedding_records(
                    relational_records,
                    vectors,
                    provider=embedding_configuration.provider,
                    model=model,
                    model_revision=model_revision,
                    dimension=embedding_configuration.expected_dimension,
                )
                assert_validation_pass(
                    validate_embedding_records(
                        embedding_records,
                        expected_count=current.expected_record_count,
                        expected_dimension=embedding_configuration.expected_dimension,
                    ),
                    stage="embedding_validation",
                )
                assert_validation_pass(
                    validate_cross_artifact_identity(
                        relational_records,
                        embedding_records,
                        expected_refs=expected_refs,
                    ),
                    stage="cross_ref_validation",
                )
                assert_validation_pass(
                    validate_semantic_text_hashes(relational_records, embedding_records),
                    stage="semantic_text_hash_validation",
                )

                current = transition_shard(current, DataPackShardStatus.WRITING)
                state = replace_shard(state, current)
                state = _persist_state(paths, state)

                relational_rel, embedding_rel = shard_descriptor_paths(current.ordinal)
                relational_path = _write_shard_atomically(
                    paths.relational_dir,
                    current.ordinal,
                    lambda temp_path: write_relational_parquet(temp_path, relational_records),
                )
                embedding_path = _write_shard_atomically(
                    paths.embeddings_dir,
                    current.ordinal,
                    lambda temp_path: write_embedding_parquet(
                        temp_path,
                        embedding_records,
                        embedding_dimension=embedding_configuration.expected_dimension,
                    ),
                )

                current = transition_shard(
                    current,
                    DataPackShardStatus.VALIDATING,
                    relational_relative_path=relational_rel,
                    embedding_relative_path=embedding_rel,
                )
                state = replace_shard(state, current)
                state = _persist_state(paths, state)

                relational_sha = sha256_file(relational_path)
                embedding_sha = sha256_file(embedding_path)
                relational_digest = source_ref_set_sha256(
                    tuple(record.source_ref for record in relational_records)
                )
                embedding_digest = source_ref_set_sha256(
                    tuple(record.source_ref for record in embedding_records)
                )
                if relational_digest != embedding_digest:
                    raise VpiDataPackBuildError(
                        f"shard {current.ordinal} relational/embedding source-ref digest mismatch"
                    )

                current = transition_shard(
                    current,
                    DataPackShardStatus.READY,
                    relational_sha256=relational_sha,
                    embedding_sha256=embedding_sha,
                    relational_source_ref_set_sha256=relational_digest,
                    embedding_source_ref_set_sha256=embedding_digest,
                )
                state = replace_shard(state, current)
                state = _persist_state(paths, state)

                relational_bytes.append(relational_path.stat().st_size)
                embedding_bytes.append(embedding_path.stat().st_size)
                logger.info(
                    "shard READY ordinal=%s records=%s",
                    current.ordinal,
                    current.expected_record_count,
                )
            except KeyboardInterrupt:
                state = _persist_state(paths, state)
                raise
            except (VpiDataPackBuildError, VpiDataPackBuildStateError) as exc:
                state = _record_failure(
                    state,
                    current,
                    error_code=exc.__class__.__name__,
                    message=str(exc),
                )
                state = _persist_state(paths, state)
                raise

        progress = compute_build_progress(state)
        all_ready = progress.ready_shards == progress.total_shards
        partial = shard_limit is not None and shard_limit < state.shard_count
        if all_ready and not partial:
            manifest = _finalize_data_pack(
                paths=paths,
                state=state,
                dataset_identity=dataset_identity,
                embedding_identity=embedding_identity,
                execution_provenance=execution_provenance,
            )
            finalized = True
    finally:
        if owns_embedding_port:
            active_embedding_port.close()

    elapsed = max(time.perf_counter() - started, 1e-9)
    embedding_elapsed = max(time.perf_counter() - embedding_started, 1e-9)
    avg_rel = int(sum(relational_bytes) / len(relational_bytes)) if relational_bytes else None
    avg_emb = int(sum(embedding_bytes) / len(embedding_bytes)) if embedding_bytes else None

    progress = compute_build_progress(state)

    return DataPackBuildReport(
        status=DataPackStatus.READY if finalized else DataPackStatus.BUILDING,
        manifest=manifest,
        progress=progress,
        records_per_second=records_embedded / elapsed if records_embedded else None,
        embedding_records_per_second=records_embedded / embedding_elapsed if records_embedded else None,
        average_relational_shard_bytes=avg_rel,
        average_embedding_shard_bytes=avg_emb,
        finalized=finalized,
    )
