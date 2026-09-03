"""Embedding materialization configuration — scenario-owned, provider-neutral."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    DatasetVerificationMode,
    DEFAULT_CATALOG_ID,
)

VPI_EMBEDDING_ARTIFACT_ENV_PREFIX = "VPI_EMBEDDING_ARTIFACT"
VPI_EMBEDDING_MATERIALIZATION_ENV_PREFIX = "VPI_EMBEDDING_MATERIALIZATION"

DEFAULT_MAX_RECORDS = 1000
DEFAULT_SOURCE_READ_BATCH_SIZE = 256
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_ARTIFACT_SHARD_SIZE = 10_000


@dataclass(frozen=True, slots=True)
class VpiEmbeddingMaterializationConfig:
    dataset_path: Path
    dataset_manifest_path: Path | None
    dataset_verification_mode: DatasetVerificationMode
    catalog_id: str
    source_revision: str | None
    max_records: int | None
    source_read_batch_size: int
    embedding_batch_size: int
    artifact_shard_size: int
    artifact_root_dir: Path
    embedding_configuration: VpiEmbeddingConfiguration

    def __post_init__(self) -> None:
        if self.source_read_batch_size <= 0:
            msg = "source_read_batch_size must be > 0"
            raise ValueError(msg)
        if self.embedding_batch_size <= 0:
            msg = "embedding_batch_size must be > 0"
            raise ValueError(msg)
        if self.artifact_shard_size <= 0:
            msg = "artifact_shard_size must be > 0"
            raise ValueError(msg)
        if self.max_records is not None and self.max_records <= 0:
            msg = "max_records must be > 0 when set"
            raise ValueError(msg)


def _scenario_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_optional_positive_int(raw_value: str | None) -> int | None:
    if raw_value is None or not raw_value.strip():
        return None
    parsed = int(raw_value.strip())
    if parsed <= 0:
        msg = "max_records must be a positive integer"
        raise ValueError(msg)
    return parsed


def load_vpi_embedding_materialization_config(
    *,
    max_records_override: int | None = None,
    artifact_dir_override: Path | None = None,
) -> VpiEmbeddingMaterializationConfig:
    scenario_root = _scenario_root()
    dataset_path = Path(
        os.getenv(
            "VPI_BOOTSTRAP_DATASET_PATH",
            str(scenario_root / "dataset" / "processed" / "selected_offers.parquet"),
        )
    )
    manifest_env = os.getenv("VPI_BOOTSTRAP_DATASET_MANIFEST_PATH", "").strip()
    default_manifest = scenario_root / "dataset" / "processed" / "selected_offers_manifest.json"
    dataset_manifest_path = Path(manifest_env) if manifest_env else default_manifest
    if not dataset_manifest_path.is_file():
        dataset_manifest_path = None

    verification_raw = os.getenv(
        "VPI_BOOTSTRAP_DATASET_VERIFICATION",
        DatasetVerificationMode.FAST.value,
    ).strip().lower()
    verification_mode = DatasetVerificationMode(verification_raw)

    max_records_env = os.getenv(
        f"{VPI_EMBEDDING_MATERIALIZATION_ENV_PREFIX}_MAX_RECORDS",
        str(DEFAULT_MAX_RECORDS),
    ).strip()
    max_records = _parse_optional_positive_int(max_records_env)
    if max_records_override is not None:
        max_records = max_records_override

    source_read_batch_size = int(
        os.getenv(
            f"{VPI_EMBEDDING_MATERIALIZATION_ENV_PREFIX}_SOURCE_READ_BATCH_SIZE",
            str(DEFAULT_SOURCE_READ_BATCH_SIZE),
        )
    )
    embedding_batch_size = int(
        os.getenv(
            f"{VPI_EMBEDDING_MATERIALIZATION_ENV_PREFIX}_BATCH_SIZE",
            str(DEFAULT_EMBEDDING_BATCH_SIZE),
        )
    )
    artifact_shard_size = int(
        os.getenv(
            f"{VPI_EMBEDDING_ARTIFACT_ENV_PREFIX}_SHARD_SIZE",
            str(DEFAULT_ARTIFACT_SHARD_SIZE),
        )
    )
    artifact_root = artifact_dir_override or Path(
        os.getenv(
            f"{VPI_EMBEDDING_ARTIFACT_ENV_PREFIX}_PATH",
            str(scenario_root / "dataset" / "processed" / "embedding_artifacts"),
        )
    )

    return VpiEmbeddingMaterializationConfig(
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        dataset_verification_mode=verification_mode,
        catalog_id=os.getenv("VPI_BOOTSTRAP_CATALOG_ID", DEFAULT_CATALOG_ID),
        source_revision=os.getenv("VPI_BOOTSTRAP_SOURCE_REVISION") or None,
        max_records=max_records,
        source_read_batch_size=source_read_batch_size,
        embedding_batch_size=embedding_batch_size,
        artifact_shard_size=artifact_shard_size,
        artifact_root_dir=artifact_root,
        embedding_configuration=load_vpi_embedding_configuration(),
    )
