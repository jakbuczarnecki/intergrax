"""JSON manifest persistence for filesystem embedding artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactIntegrityError,
    ArtifactWriteError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
    EmbeddingArtifactShardDescriptor,
    EmbeddingArtifactState,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class EmbeddingArtifactShardManifestPayload(TypedDict):
    shard_ordinal: int
    file_name: str
    first_global_row_index: int
    last_global_row_index: int
    record_count: int
    sha256_checksum: str


class EmbeddingArtifactManifestPayload(TypedDict):
    state: str
    artifact_schema_version: str
    dataset_path: str
    dataset_checksum: str
    dataset_record_count: int
    search_representation_derivation_version: str
    embedding_configuration_version: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    catalog_id: str
    source_revision: str | None
    checkpoint_shard_ordinal: int | None
    checkpoint_rows_materialized: int
    target_max_records: int | None
    total_artifact_record_count: int
    shard_count: int
    committed_shards: list[EmbeddingArtifactShardManifestPayload]
    created_at_utc: str | None
    finalized_at_utc: str | None
    failure_stage: str | None
    failure_detail: str | None


def _integrity(message: str) -> ArtifactIntegrityError:
    return ArtifactIntegrityError(message)


def _require_str_field(payload: Mapping[str, JsonValue], field: str) -> str:
    if field not in payload:
        raise _integrity(f"manifest field missing: {field}")
    value = payload[field]
    if not isinstance(value, str):
        raise _integrity(f"manifest field must be string: {field}")
    return value


def _require_int_field(payload: Mapping[str, JsonValue], field: str) -> int:
    if field not in payload:
        raise _integrity(f"manifest field missing: {field}")
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _integrity(f"manifest field must be integer: {field}")
    return value


def _optional_str_field(payload: Mapping[str, JsonValue], field: str) -> str | None:
    if field not in payload:
        return None
    value = payload[field]
    if value is None:
        return None
    if not isinstance(value, str):
        raise _integrity(f"manifest field must be string or null: {field}")
    return value


def _optional_int_field(payload: Mapping[str, JsonValue], field: str) -> int | None:
    if field not in payload:
        return None
    value = payload[field]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _integrity(f"manifest field must be integer or null: {field}")
    return value


def _require_non_negative_int(payload: Mapping[str, JsonValue], field: str) -> int:
    value = _require_int_field(payload, field)
    if value < 0:
        raise _integrity(f"manifest field must be non-negative: {field}")
    return value


def _require_positive_int(payload: Mapping[str, JsonValue], field: str) -> int:
    value = _require_int_field(payload, field)
    if value <= 0:
        raise _integrity(f"manifest field must be positive: {field}")
    return value


def _shard_descriptor_to_payload(
    descriptor: EmbeddingArtifactShardDescriptor,
) -> EmbeddingArtifactShardManifestPayload:
    return EmbeddingArtifactShardManifestPayload(
        shard_ordinal=descriptor.shard_ordinal,
        file_name=descriptor.file_name,
        first_global_row_index=descriptor.first_global_row_index,
        last_global_row_index=descriptor.last_global_row_index,
        record_count=descriptor.record_count,
        sha256_checksum=descriptor.sha256_checksum,
    )


def _shard_descriptor_from_payload(
    payload: Mapping[str, JsonValue],
) -> EmbeddingArtifactShardDescriptor:
    shard_ordinal = _require_non_negative_int(payload, "shard_ordinal")
    file_name = _require_str_field(payload, "file_name")
    first_global_row_index = _require_non_negative_int(payload, "first_global_row_index")
    last_global_row_index = _require_non_negative_int(payload, "last_global_row_index")
    record_count = _require_positive_int(payload, "record_count")
    sha256_checksum = _require_str_field(payload, "sha256_checksum")
    if last_global_row_index < first_global_row_index:
        raise _integrity("manifest shard last_global_row_index must be >= first_global_row_index")
    return EmbeddingArtifactShardDescriptor(
        shard_ordinal=shard_ordinal,
        file_name=file_name,
        first_global_row_index=first_global_row_index,
        last_global_row_index=last_global_row_index,
        record_count=record_count,
        sha256_checksum=sha256_checksum,
    )


def manifest_to_dict(manifest: EmbeddingArtifactManifest) -> EmbeddingArtifactManifestPayload:
    return EmbeddingArtifactManifestPayload(
        state=manifest.state.value,
        artifact_schema_version=manifest.artifact_schema_version,
        dataset_path=manifest.dataset_path,
        dataset_checksum=manifest.dataset_checksum,
        dataset_record_count=manifest.dataset_record_count,
        search_representation_derivation_version=manifest.search_representation_derivation_version,
        embedding_configuration_version=manifest.embedding_configuration_version,
        embedding_provider=manifest.embedding_provider,
        embedding_model=manifest.embedding_model,
        embedding_dimension=manifest.embedding_dimension,
        catalog_id=manifest.catalog_id,
        source_revision=manifest.source_revision,
        checkpoint_shard_ordinal=manifest.checkpoint_shard_ordinal,
        checkpoint_rows_materialized=manifest.checkpoint_rows_materialized,
        target_max_records=manifest.target_max_records,
        total_artifact_record_count=manifest.total_artifact_record_count,
        shard_count=manifest.shard_count,
        committed_shards=[
            _shard_descriptor_to_payload(descriptor) for descriptor in manifest.committed_shards
        ],
        created_at_utc=manifest.created_at_utc,
        finalized_at_utc=manifest.finalized_at_utc,
        failure_stage=manifest.failure_stage,
        failure_detail=manifest.failure_detail,
    )


def manifest_from_dict(payload: Mapping[str, JsonValue]) -> EmbeddingArtifactManifest:
    state = _require_str_field(payload, "state")
    artifact_schema_version = _require_str_field(payload, "artifact_schema_version")
    dataset_path = _require_str_field(payload, "dataset_path")
    dataset_checksum = _require_str_field(payload, "dataset_checksum")
    dataset_record_count = _require_non_negative_int(payload, "dataset_record_count")
    search_representation_derivation_version = _require_str_field(
        payload,
        "search_representation_derivation_version",
    )
    embedding_configuration_version = _require_str_field(
        payload,
        "embedding_configuration_version",
    )
    embedding_provider = _require_str_field(payload, "embedding_provider")
    embedding_model = _require_str_field(payload, "embedding_model")
    embedding_dimension = _require_positive_int(payload, "embedding_dimension")
    catalog_id = _require_str_field(payload, "catalog_id")
    source_revision = _optional_str_field(payload, "source_revision")
    checkpoint_shard_ordinal = _optional_int_field(payload, "checkpoint_shard_ordinal")
    if checkpoint_shard_ordinal is not None and checkpoint_shard_ordinal < 0:
        raise _integrity("manifest field must be non-negative: checkpoint_shard_ordinal")
    checkpoint_rows_materialized = _require_non_negative_int(
        payload,
        "checkpoint_rows_materialized",
    )
    target_max_records = _optional_int_field(payload, "target_max_records")
    if target_max_records is not None and target_max_records < 0:
        raise _integrity("manifest field must be non-negative: target_max_records")
    total_artifact_record_count = _require_non_negative_int(
        payload,
        "total_artifact_record_count",
    )
    shard_count = _require_non_negative_int(payload, "shard_count")
    committed_raw = payload.get("committed_shards")
    if committed_raw is None:
        raise _integrity("manifest field missing: committed_shards")
    if not isinstance(committed_raw, list):
        raise _integrity("manifest committed_shards must be a list")
    committed_shards: list[EmbeddingArtifactShardDescriptor] = []
    for index, item in enumerate(committed_raw):
        if not isinstance(item, dict):
            raise _integrity(f"manifest shard descriptor must be an object at index {index}")
        committed_shards.append(_shard_descriptor_from_payload(item))
    try:
        artifact_state = EmbeddingArtifactState(state)
    except ValueError as exc:
        raise _integrity(f"manifest field has invalid state: state") from exc
    return EmbeddingArtifactManifest(
        state=artifact_state,
        artifact_schema_version=artifact_schema_version,
        dataset_path=dataset_path,
        dataset_checksum=dataset_checksum,
        dataset_record_count=dataset_record_count,
        search_representation_derivation_version=search_representation_derivation_version,
        embedding_configuration_version=embedding_configuration_version,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        catalog_id=catalog_id,
        source_revision=source_revision,
        checkpoint_shard_ordinal=checkpoint_shard_ordinal,
        checkpoint_rows_materialized=checkpoint_rows_materialized,
        target_max_records=target_max_records,
        total_artifact_record_count=total_artifact_record_count,
        shard_count=shard_count,
        committed_shards=tuple(committed_shards),
        created_at_utc=_optional_str_field(payload, "created_at_utc"),
        finalized_at_utc=_optional_str_field(payload, "finalized_at_utc"),
        failure_stage=_optional_str_field(payload, "failure_stage"),
        failure_detail=_optional_str_field(payload, "failure_detail"),
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
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"failed to read manifest: {path}") from exc
    if not isinstance(raw_payload, dict):
        raise ArtifactIntegrityError("manifest must be a JSON object")
    payload = cast(dict[str, JsonValue], raw_payload)
    return manifest_from_dict(payload)
