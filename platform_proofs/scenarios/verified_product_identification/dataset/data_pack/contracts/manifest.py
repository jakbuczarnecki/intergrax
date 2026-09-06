"""Typed manifest and shard descriptors for universal data packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackCompatibilityError,
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_FORMAT,
    RELATIONAL_FORMAT,
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


@dataclass(frozen=True, slots=True)
class SampleIdentity:
    sample_version: str
    sample_seed: int
    selected_record_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingPackIdentity:
    provider: str
    model: str
    model_revision: str | None
    dimension: int
    embedding_configuration_version: str
    input_policy_version: str
    execution_configuration_identity: str


@dataclass(frozen=True, slots=True)
class DataPackShardDescriptor:
    shard_ordinal: int
    file_name: str
    record_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DataPackManifest:
    data_pack_version: str
    scenario_id: str
    source_dataset: SourceDatasetIdentity
    source_record_count: int
    sample_identity: SampleIdentity
    derivation_version: str
    semantic_text_version: str
    embedding_identity: EmbeddingPackIdentity
    relational_format: str
    embedding_format: str
    shard_count: int
    record_count: int
    created_at_utc: str
    status: DataPackStatus
    checksums_path: str
    shards_index_path: str
    relational_shard_file: str
    embedding_shard_file: str

    def __post_init__(self) -> None:
        if self.record_count <= 0:
            raise ValueError("record_count must be > 0")
        if self.shard_count <= 0:
            raise ValueError("shard_count must be > 0")
        if self.source_record_count != self.record_count:
            raise ValueError("source_record_count must equal record_count for proof packs")


def manifest_to_json_dict(manifest: DataPackManifest) -> dict[str, object]:
    return {
        "data_pack_version": manifest.data_pack_version,
        "scenario_id": manifest.scenario_id,
        "source_dataset": {
            "dataset_name": manifest.source_dataset.dataset_name,
            "dataset_path": manifest.source_dataset.dataset_path,
            "dataset_sha256": manifest.source_dataset.dataset_sha256,
            "dataset_record_count": manifest.source_dataset.dataset_record_count,
        },
        "source_record_count": manifest.source_record_count,
        "sample_identity": {
            "sample_version": manifest.sample_identity.sample_version,
            "sample_seed": manifest.sample_identity.sample_seed,
            "selected_record_refs": list(manifest.sample_identity.selected_record_refs),
        },
        "derivation_version": manifest.derivation_version,
        "semantic_text_version": manifest.semantic_text_version,
        "embedding_identity": {
            "provider": manifest.embedding_identity.provider,
            "model": manifest.embedding_identity.model,
            "model_revision": manifest.embedding_identity.model_revision,
            "dimension": manifest.embedding_identity.dimension,
            "embedding_configuration_version": (
                manifest.embedding_identity.embedding_configuration_version
            ),
            "input_policy_version": manifest.embedding_identity.input_policy_version,
            "execution_configuration_identity": (
                manifest.embedding_identity.execution_configuration_identity
            ),
        },
        "relational_format": manifest.relational_format,
        "embedding_format": manifest.embedding_format,
        "shard_count": manifest.shard_count,
        "record_count": manifest.record_count,
        "created_at_utc": manifest.created_at_utc,
        "status": manifest.status.value,
        "checksums_path": manifest.checksums_path,
        "shards_index_path": manifest.shards_index_path,
        "relational_shard_file": manifest.relational_shard_file,
        "embedding_shard_file": manifest.embedding_shard_file,
    }


def manifest_from_json_dict(payload: dict[str, object]) -> DataPackManifest:
    source_raw = payload.get("source_dataset")
    if not isinstance(source_raw, dict):
        raise VpiDataPackCompatibilityError("manifest.source_dataset must be an object")
    sample_raw = payload.get("sample_identity")
    if not isinstance(sample_raw, dict):
        raise VpiDataPackCompatibilityError("manifest.sample_identity must be an object")
    embedding_raw = payload.get("embedding_identity")
    if not isinstance(embedding_raw, dict):
        raise VpiDataPackCompatibilityError("manifest.embedding_identity must be an object")

    selected_refs_raw = sample_raw.get("selected_record_refs")
    if not isinstance(selected_refs_raw, list):
        raise VpiDataPackCompatibilityError("sample_identity.selected_record_refs must be a list")
    selected_refs = tuple(str(item) for item in selected_refs_raw)

    status_raw = str(payload.get("status", ""))
    try:
        status = DataPackStatus(status_raw)
    except ValueError as exc:
        raise VpiDataPackCompatibilityError(f"unsupported manifest status: {status_raw}") from exc

    model_revision_raw = embedding_raw.get("model_revision")
    model_revision = str(model_revision_raw) if model_revision_raw is not None else None

    return DataPackManifest(
        data_pack_version=str(payload.get("data_pack_version", "")),
        scenario_id=str(payload.get("scenario_id", "")),
        source_dataset=SourceDatasetIdentity(
            dataset_name=str(source_raw.get("dataset_name", "")),
            dataset_path=str(source_raw.get("dataset_path", "")),
            dataset_sha256=str(source_raw.get("dataset_sha256", "")),
            dataset_record_count=int(source_raw.get("dataset_record_count", 0)),
        ),
        source_record_count=int(payload.get("source_record_count", 0)),
        sample_identity=SampleIdentity(
            sample_version=str(sample_raw.get("sample_version", "")),
            sample_seed=int(sample_raw.get("sample_seed", 0)),
            selected_record_refs=selected_refs,
        ),
        derivation_version=str(payload.get("derivation_version", "")),
        semantic_text_version=str(payload.get("semantic_text_version", "")),
        embedding_identity=EmbeddingPackIdentity(
            provider=str(embedding_raw.get("provider", "")),
            model=str(embedding_raw.get("model", "")),
            model_revision=model_revision,
            dimension=int(embedding_raw.get("dimension", 0)),
            embedding_configuration_version=str(
                embedding_raw.get("embedding_configuration_version", "")
            ),
            input_policy_version=str(embedding_raw.get("input_policy_version", "")),
            execution_configuration_identity=str(
                embedding_raw.get("execution_configuration_identity", "")
            ),
        ),
        relational_format=str(payload.get("relational_format", RELATIONAL_FORMAT)),
        embedding_format=str(payload.get("embedding_format", EMBEDDING_FORMAT)),
        shard_count=int(payload.get("shard_count", 0)),
        record_count=int(payload.get("record_count", 0)),
        created_at_utc=str(payload.get("created_at_utc", "")),
        status=status,
        checksums_path=str(payload.get("checksums_path", "")),
        shards_index_path=str(payload.get("shards_index_path", "")),
        relational_shard_file=str(payload.get("relational_shard_file", "")),
        embedding_shard_file=str(payload.get("embedding_shard_file", "")),
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
        raise VpiDataPackCompatibilityError("manifest root must be a JSON object")
    return manifest_from_json_dict(payload)
