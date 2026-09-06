"""Typed manifest and shard descriptors for universal data packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackCompatibilityError,
    VpiDataPackFormatError,
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    PARQUET_FILE_FORMAT,
    RELATIONAL_SCHEMA_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.json_decode import (
    require_int,
    require_mapping,
    require_optional_str,
    require_str,
    require_str_list,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)


@dataclass(frozen=True, slots=True)
class SourceDatasetIdentity:
    dataset_name: str
    dataset_path: str
    dataset_sha256: str
    dataset_record_count: int

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must be non-empty")
        if not self.dataset_sha256.strip():
            raise ValueError("dataset_sha256 must be non-empty")
        if self.dataset_record_count <= 0:
            raise ValueError("dataset_record_count must be > 0")


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    """Optional proof/sample metadata; absent for full production packs."""

    sample_version: str
    sample_seed: int
    selected_record_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuildExecutionProvenance:
    """Non-semantic execution metadata; excluded from content identity."""

    device: str | None
    provider_batch_size: int | None


@dataclass(frozen=True, slots=True)
class EmbeddingPackIdentity:
    provider: str
    model: str
    model_revision: str | None
    artifact_fingerprint: str | None
    dimension: int
    embedding_configuration_version: str
    input_policy_version: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be > 0")
        if self.model_revision is None and self.artifact_fingerprint is None:
            raise ValueError("model_revision or artifact_fingerprint is required")

    def resolved_model_identity(self) -> str:
        if self.model_revision is not None:
            return self.model_revision
        if self.artifact_fingerprint is not None:
            return self.artifact_fingerprint
        raise ValueError("embedding model identity is unresolved")


@dataclass(frozen=True, slots=True)
class DataPackManifest:
    data_pack_version: str
    content_identity: str
    scenario_id: str
    source_dataset: SourceDatasetIdentity
    source_record_count: int
    sample_identity: SampleIdentity | None
    derivation_version: str
    semantic_text_version: str
    embedding_identity: EmbeddingPackIdentity
    relational_schema_version: str
    embedding_schema_version: str
    relational_format: str
    embedding_format: str
    shard_count: int
    record_count: int
    created_at_utc: str
    status: DataPackStatus
    checksums_path: str
    shards_index_path: str
    build_execution_provenance: BuildExecutionProvenance | None

    def __post_init__(self) -> None:
        if self.record_count <= 0:
            raise ValueError("record_count must be > 0")
        if self.shard_count <= 0:
            raise ValueError("shard_count must be > 0")
        if self.source_record_count != self.record_count:
            raise ValueError("source_record_count must equal record_count")
        if not self.content_identity.strip():
            raise ValueError("content_identity must be non-empty")


def _source_dataset_to_dict(source_dataset: SourceDatasetIdentity) -> dict[str, object]:
    return {
        "dataset_name": source_dataset.dataset_name,
        "dataset_path": source_dataset.dataset_path,
        "dataset_sha256": source_dataset.dataset_sha256,
        "dataset_record_count": source_dataset.dataset_record_count,
    }


def _embedding_identity_to_dict(embedding_identity: EmbeddingPackIdentity) -> dict[str, object]:
    return {
        "provider": embedding_identity.provider,
        "model": embedding_identity.model,
        "model_revision": embedding_identity.model_revision,
        "artifact_fingerprint": embedding_identity.artifact_fingerprint,
        "dimension": embedding_identity.dimension,
        "embedding_configuration_version": embedding_identity.embedding_configuration_version,
        "input_policy_version": embedding_identity.input_policy_version,
    }


def manifest_to_json_dict(manifest: DataPackManifest) -> dict[str, object]:
    sample_identity: dict[str, object] | None = None
    if manifest.sample_identity is not None:
        sample_identity = {
            "sample_version": manifest.sample_identity.sample_version,
            "sample_seed": manifest.sample_identity.sample_seed,
            "selected_record_refs": list(manifest.sample_identity.selected_record_refs),
        }
    build_execution_provenance: dict[str, object] | None = None
    if manifest.build_execution_provenance is not None:
        build_execution_provenance = {
            "device": manifest.build_execution_provenance.device,
            "provider_batch_size": manifest.build_execution_provenance.provider_batch_size,
        }
    return {
        "data_pack_version": manifest.data_pack_version,
        "content_identity": manifest.content_identity,
        "scenario_id": manifest.scenario_id,
        "source_dataset": _source_dataset_to_dict(manifest.source_dataset),
        "source_record_count": manifest.source_record_count,
        "sample_identity": sample_identity,
        "derivation_version": manifest.derivation_version,
        "semantic_text_version": manifest.semantic_text_version,
        "embedding_identity": _embedding_identity_to_dict(manifest.embedding_identity),
        "relational_schema_version": manifest.relational_schema_version,
        "embedding_schema_version": manifest.embedding_schema_version,
        "relational_format": manifest.relational_format,
        "embedding_format": manifest.embedding_format,
        "shard_count": manifest.shard_count,
        "record_count": manifest.record_count,
        "created_at_utc": manifest.created_at_utc,
        "status": manifest.status.value,
        "checksums_path": manifest.checksums_path,
        "shards_index_path": manifest.shards_index_path,
        "build_execution_provenance": build_execution_provenance,
    }


def _source_dataset_from_dict(payload: dict[str, object]) -> SourceDatasetIdentity:
    return SourceDatasetIdentity(
        dataset_name=require_str(payload, "dataset_name"),
        dataset_path=require_str(payload, "dataset_path"),
        dataset_sha256=require_str(payload, "dataset_sha256"),
        dataset_record_count=require_int(payload, "dataset_record_count", minimum=1),
    )


def _embedding_identity_from_dict(payload: dict[str, object]) -> EmbeddingPackIdentity:
    return EmbeddingPackIdentity(
        provider=require_str(payload, "provider"),
        model=require_str(payload, "model"),
        model_revision=require_optional_str(payload, "model_revision"),
        artifact_fingerprint=require_optional_str(payload, "artifact_fingerprint"),
        dimension=require_int(payload, "dimension", minimum=1),
        embedding_configuration_version=require_str(payload, "embedding_configuration_version"),
        input_policy_version=require_str(payload, "input_policy_version"),
    )


def _sample_identity_from_dict(payload: dict[str, object] | None) -> SampleIdentity | None:
    if payload is None:
        return None
    return SampleIdentity(
        sample_version=require_str(payload, "sample_version"),
        sample_seed=require_int(payload, "sample_seed", minimum=0),
        selected_record_refs=require_str_list(payload, "selected_record_refs"),
    )


def _build_execution_provenance_from_dict(
    payload: dict[str, object] | None,
) -> BuildExecutionProvenance | None:
    if payload is None:
        return None
    device_raw = payload.get("device")
    device = None if device_raw is None else require_str(payload, "device")
    batch_raw = payload.get("provider_batch_size")
    provider_batch_size = None
    if batch_raw is not None:
        provider_batch_size = require_int(payload, "provider_batch_size", minimum=1)
    return BuildExecutionProvenance(device=device, provider_batch_size=provider_batch_size)


def manifest_from_json_dict(payload: dict[str, object]) -> DataPackManifest:
    source_raw = payload.get("source_dataset")
    embedding_raw = payload.get("embedding_identity")
    source_dataset = _source_dataset_from_dict(require_mapping(source_raw, field_name="source_dataset"))
    embedding_identity = _embedding_identity_from_dict(
        require_mapping(embedding_raw, field_name="embedding_identity")
    )
    sample_raw = payload.get("sample_identity")
    sample_identity = _sample_identity_from_dict(
        require_mapping(sample_raw, field_name="sample_identity") if sample_raw is not None else None
    )
    provenance_raw = payload.get("build_execution_provenance")
    build_execution_provenance = _build_execution_provenance_from_dict(
        require_mapping(provenance_raw, field_name="build_execution_provenance")
        if provenance_raw is not None
        else None
    )

    status_raw = payload.get("status")
    if not isinstance(status_raw, str):
        raise VpiDataPackFormatError("status must be a string")
    try:
        status = DataPackStatus(status_raw)
    except ValueError as exc:
        raise VpiDataPackCompatibilityError(f"unsupported manifest status: {status_raw}") from exc

    return DataPackManifest(
        data_pack_version=require_str(payload, "data_pack_version"),
        content_identity=require_str(payload, "content_identity"),
        scenario_id=require_str(payload, "scenario_id"),
        source_dataset=source_dataset,
        source_record_count=require_int(payload, "source_record_count", minimum=1),
        sample_identity=sample_identity,
        derivation_version=require_str(payload, "derivation_version"),
        semantic_text_version=require_str(payload, "semantic_text_version"),
        embedding_identity=embedding_identity,
        relational_schema_version=require_str(payload, "relational_schema_version"),
        embedding_schema_version=require_str(payload, "embedding_schema_version"),
        relational_format=require_str(payload, "relational_format"),
        embedding_format=require_str(payload, "embedding_format"),
        shard_count=require_int(payload, "shard_count", minimum=1),
        record_count=require_int(payload, "record_count", minimum=1),
        created_at_utc=require_str(payload, "created_at_utc"),
        status=status,
        checksums_path=require_str(payload, "checksums_path"),
        shards_index_path=require_str(payload, "shards_index_path"),
        build_execution_provenance=build_execution_provenance,
    )


def write_manifest_file(path: Path, manifest: DataPackManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(manifest_to_json_dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def read_manifest_file(path: Path) -> DataPackManifest:
    if not path.is_file():
        raise VpiDataPackIntegrityError(f"manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VpiDataPackFormatError("manifest root must be a JSON object")
    return manifest_from_json_dict(payload)
