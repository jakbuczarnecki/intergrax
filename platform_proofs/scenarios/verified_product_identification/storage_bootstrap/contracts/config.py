"""Bootstrap configuration — scenario-owned, provider-neutral."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    CATALOG_SCHEMA_VERSION,
    SEARCH_INDEX_SCHEMA_VERSION,
)

VPI_BOOTSTRAP_ENV_PREFIX = "VPI_BOOTSTRAP"
VPI_EMBEDDING_ARTIFACT_ENV_PREFIX = "VPI_EMBEDDING_ARTIFACT"
VPI_DATA_PACKAGE_ENV_PREFIX = "VPI_DATA_PACKAGE"
DEFAULT_CATALOG_ID = "wdc-v2-selected"
DEFAULT_SOURCE_BATCH_SIZE = 256
DEFAULT_VECTOR_BATCH_SIZE = 64
DEFAULT_POSTGRESQL_SCHEMA = "vpi"
DEFAULT_QDRANT_COLLECTION = "vpi_offers"


class DatasetVerificationMode(str, Enum):
    FAST = "fast"
    FULL = "full"


class BootstrapRunMode(str, Enum):
    VERIFY = "verify"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class VpiBootstrapConfig:
    dataset_path: Path
    dataset_manifest_path: Path | None
    dataset_verification_mode: DatasetVerificationMode
    catalog_id: str
    source_revision: str | None
    max_records: int | None
    source_batch_size: int
    vector_batch_size: int
    catalog_schema_version: str
    search_index_schema_version: str
    bootstrap_implementation_version: str
    postgresql_schema: str
    qdrant_collection_name: str
    artifact_root_dir: Path
    embedding_configuration: VpiEmbeddingConfiguration

    def __post_init__(self) -> None:
        if self.source_batch_size <= 0:
            msg = "source_batch_size must be > 0"
            raise ValueError(msg)
        if self.vector_batch_size <= 0:
            msg = "vector_batch_size must be > 0"
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


def _resolve_data_paths_from_installed_package(
    scenario_root: Path,
) -> tuple[Path, Path | None, Path] | None:
    from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
        VpiDataPackageNotInstalledError,
    )
    from platform_proofs.scenarios.verified_product_identification.data_package.paths import (
        assert_installed_data_present,
        resolve_installed_data_paths,
    )

    explicit_install_raw = os.getenv(f"{VPI_DATA_PACKAGE_ENV_PREFIX}_INSTALL_DIR", "").strip()
    if explicit_install_raw:
        paths = resolve_installed_data_paths(Path(explicit_install_raw))
        assert_installed_data_present(paths)
        return paths.dataset_path, paths.dataset_manifest_path, paths.embedding_artifact_root

    from platform_proofs.scenarios.verified_product_identification.data_package.config import (
        default_install_dir,
    )

    install_root = default_install_dir()
    if not install_root.is_dir():
        return None

    paths = resolve_installed_data_paths(install_root)
    try:
        assert_installed_data_present(paths)
    except VpiDataPackageNotInstalledError:
        return None
    return paths.dataset_path, paths.dataset_manifest_path, paths.embedding_artifact_root


def load_vpi_bootstrap_config(
    *,
    mode: BootstrapRunMode,
    max_records_override: int | None = None,
) -> VpiBootstrapConfig:
    scenario_root = _scenario_root()
    installed_paths = _resolve_data_paths_from_installed_package(scenario_root)
    if installed_paths is not None:
        dataset_path, dataset_manifest_path, artifact_root_dir = installed_paths
    else:
        dataset_path = Path(
            os.getenv(
                f"{VPI_BOOTSTRAP_ENV_PREFIX}_DATASET_PATH",
                str(scenario_root / "dataset" / "processed" / "selected_offers.parquet"),
            )
        )
        manifest_env = os.getenv(f"{VPI_BOOTSTRAP_ENV_PREFIX}_DATASET_MANIFEST_PATH", "").strip()
        default_manifest = scenario_root / "dataset" / "processed" / "selected_offers_manifest.json"
        dataset_manifest_path = Path(manifest_env) if manifest_env else default_manifest
        if not dataset_manifest_path.is_file():
            dataset_manifest_path = None
        artifact_root_dir = Path(
            os.getenv(
                f"{VPI_EMBEDDING_ARTIFACT_ENV_PREFIX}_PATH",
                str(scenario_root / "dataset" / "processed" / "embedding_artifacts"),
            )
        )

    verification_raw = os.getenv(
        f"{VPI_BOOTSTRAP_ENV_PREFIX}_DATASET_VERIFICATION",
        DatasetVerificationMode.FAST.value,
    ).strip().lower()
    verification_mode = DatasetVerificationMode(verification_raw)

    max_records_env = os.getenv(f"{VPI_BOOTSTRAP_ENV_PREFIX}_MAX_RECORDS", "").strip()
    max_records = _parse_optional_positive_int(max_records_env)
    if mode is BootstrapRunMode.VERIFY:
        max_records = max_records_override if max_records_override is not None else (max_records or 1000)
    elif max_records_override is not None:
        max_records = max_records_override

    source_batch_size = int(
        os.getenv(
            f"{VPI_BOOTSTRAP_ENV_PREFIX}_SOURCE_BATCH_SIZE",
            str(DEFAULT_SOURCE_BATCH_SIZE),
        )
    )
    vector_batch_size = int(
        os.getenv(
            f"{VPI_BOOTSTRAP_ENV_PREFIX}_VECTOR_BATCH_SIZE",
            str(DEFAULT_VECTOR_BATCH_SIZE),
        )
    )
    return VpiBootstrapConfig(
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        dataset_verification_mode=verification_mode,
        catalog_id=os.getenv(f"{VPI_BOOTSTRAP_ENV_PREFIX}_CATALOG_ID", DEFAULT_CATALOG_ID),
        source_revision=os.getenv(f"{VPI_BOOTSTRAP_ENV_PREFIX}_SOURCE_REVISION") or None,
        max_records=max_records,
        source_batch_size=source_batch_size,
        vector_batch_size=vector_batch_size,
        catalog_schema_version=CATALOG_SCHEMA_VERSION,
        search_index_schema_version=SEARCH_INDEX_SCHEMA_VERSION,
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        postgresql_schema=os.getenv(
            f"{VPI_BOOTSTRAP_ENV_PREFIX}_POSTGRESQL_SCHEMA",
            DEFAULT_POSTGRESQL_SCHEMA,
        ),
        qdrant_collection_name=os.getenv(
            f"{VPI_BOOTSTRAP_ENV_PREFIX}_QDRANT_COLLECTION",
            DEFAULT_QDRANT_COLLECTION,
        ),
        artifact_root_dir=artifact_root_dir,
        embedding_configuration=load_vpi_embedding_configuration(),
    )
