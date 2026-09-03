"""JSON manifest persistence for filesystem embedding artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactIntegrityError,
    ArtifactWriteError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
    EmbeddingArtifactShardDescriptor,
    EmbeddingArtifactState,
)


def _shard_descriptor_to_dict(descriptor: EmbeddingArtifactShardDescriptor) -> dict[str, int | str]:
    return {
        "shard_ordinal": descriptor.shard_ordinal,
        "file_name": descriptor.file_name,
        "first_global_row_index": descriptor.first_global_row_index,
        "last_global_row_index": descriptor.last_global_row_index,
        "record_count": descriptor.record_count,
        "sha256_checksum": descriptor.sha256_checksum,
    }


def _shard_descriptor_from_dict(payload: dict[str, object]) -> EmbeddingArtifactShardDescriptor:
    return EmbeddingArtifactShardDescriptor(
        shard_ordinal=int(payload["shard_ordinal"]),
        file_name=str(payload["file_name"]),
        first_global_row_index=int(payload["first_global_row_index"]),
        last_global_row_index=int(payload["last_global_row_index"]),
        record_count=int(payload["record_count"]),
        sha256_checksum=str(payload["sha256_checksum"]),
    )


def manifest_to_dict(manifest: EmbeddingArtifactManifest) -> dict[str, object]:
    return {
        "state": manifest.state.value,
        "artifact_schema_version": manifest.artifact_schema_version,
        "dataset_path": manifest.dataset_path,
        "dataset_checksum": manifest.dataset_checksum,
        "dataset_record_count": manifest.dataset_record_count,
        "search_representation_derivation_version": manifest.search_representation_derivation_version,
        "embedding_configuration_version": manifest.embedding_configuration_version,
        "embedding_provider": manifest.embedding_provider,
        "embedding_model": manifest.embedding_model,
        "embedding_dimension": manifest.embedding_dimension,
        "catalog_id": manifest.catalog_id,
        "source_revision": manifest.source_revision,
        "checkpoint_shard_ordinal": manifest.checkpoint_shard_ordinal,
        "checkpoint_rows_materialized": manifest.checkpoint_rows_materialized,
        "target_max_records": manifest.target_max_records,
        "total_artifact_record_count": manifest.total_artifact_record_count,
        "shard_count": manifest.shard_count,
        "committed_shards": [
            _shard_descriptor_to_dict(descriptor) for descriptor in manifest.committed_shards
        ],
        "created_at_utc": manifest.created_at_utc,
        "finalized_at_utc": manifest.finalized_at_utc,
        "failure_stage": manifest.failure_stage,
        "failure_detail": manifest.failure_detail,
    }


def manifest_from_dict(payload: dict[str, object]) -> EmbeddingArtifactManifest:
    committed_raw = payload.get("committed_shards")
    if not isinstance(committed_raw, list):
        raise ArtifactIntegrityError("manifest committed_shards must be a list")
    committed_shards: list[EmbeddingArtifactShardDescriptor] = []
    for item in committed_raw:
        if not isinstance(item, dict):
            raise ArtifactIntegrityError("manifest shard descriptor must be an object")
        committed_shards.append(_shard_descriptor_from_dict(item))
    return EmbeddingArtifactManifest(
        state=EmbeddingArtifactState(str(payload["state"])),
        artifact_schema_version=str(payload["artifact_schema_version"]),
        dataset_path=str(payload["dataset_path"]),
        dataset_checksum=str(payload["dataset_checksum"]),
        dataset_record_count=int(payload["dataset_record_count"]),
        search_representation_derivation_version=str(
            payload["search_representation_derivation_version"]
        ),
        embedding_configuration_version=str(payload["embedding_configuration_version"]),
        embedding_provider=str(payload["embedding_provider"]),
        embedding_model=str(payload["embedding_model"]),
        embedding_dimension=int(payload["embedding_dimension"]),
        catalog_id=str(payload["catalog_id"]),
        source_revision=(
            str(payload["source_revision"])
            if payload.get("source_revision") is not None
            else None
        ),
        checkpoint_shard_ordinal=(
            int(payload["checkpoint_shard_ordinal"])
            if payload.get("checkpoint_shard_ordinal") is not None
            else None
        ),
        checkpoint_rows_materialized=int(payload["checkpoint_rows_materialized"]),
        target_max_records=(
            int(payload["target_max_records"])
            if payload.get("target_max_records") is not None
            else None
        ),
        total_artifact_record_count=int(payload["total_artifact_record_count"]),
        shard_count=int(payload["shard_count"]),
        committed_shards=tuple(committed_shards),
        created_at_utc=(
            str(payload["created_at_utc"]) if payload.get("created_at_utc") is not None else None
        ),
        finalized_at_utc=(
            str(payload["finalized_at_utc"])
            if payload.get("finalized_at_utc") is not None
            else None
        ),
        failure_stage=(
            str(payload["failure_stage"]) if payload.get("failure_stage") is not None else None
        ),
        failure_detail=(
            str(payload["failure_detail"]) if payload.get("failure_detail") is not None else None
        ),
    )


def write_manifest_file(path: Path, manifest: EmbeddingArtifactManifest) -> None:
    payload = manifest_to_dict(manifest)
    temp_path = path.with_suffix(".json.tmp")
    try:
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        raise ArtifactWriteError(f"failed to write manifest: {path}") from exc


def read_manifest_file(path: Path) -> EmbeddingArtifactManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"failed to read manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError("manifest must be a JSON object")
    return manifest_from_dict(payload)
