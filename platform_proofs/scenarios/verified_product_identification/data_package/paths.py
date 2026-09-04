"""Installed VPI data package path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageNotInstalledError,
)

DATASET_RELATIVE_PATH = Path("dataset/selected_offers.parquet")
DATASET_MANIFEST_RELATIVE_PATH = Path("dataset/manifest.json")
EMBEDDING_ROOT_RELATIVE_PATH = Path("embeddings")
EMBEDDING_MANIFEST_RELATIVE_PATH = Path("embeddings/manifest.json")
PROVENANCE_RELATIVE_PATH = Path("provenance.json")


@dataclass(frozen=True, slots=True)
class VpiInstalledDataPaths:
    install_root: Path
    dataset_path: Path
    dataset_manifest_path: Path
    embedding_artifact_root: Path
    embedding_manifest_path: Path
    provenance_path: Path


def resolve_installed_data_paths(install_root: Path) -> VpiInstalledDataPaths:
    root = install_root.resolve()
    return VpiInstalledDataPaths(
        install_root=root,
        dataset_path=root / DATASET_RELATIVE_PATH,
        dataset_manifest_path=root / DATASET_MANIFEST_RELATIVE_PATH,
        embedding_artifact_root=root / EMBEDDING_ROOT_RELATIVE_PATH,
        embedding_manifest_path=root / EMBEDDING_MANIFEST_RELATIVE_PATH,
        provenance_path=root / PROVENANCE_RELATIVE_PATH,
    )


def assert_installed_data_present(paths: VpiInstalledDataPaths) -> None:
    missing: list[str] = []
    if not paths.dataset_path.is_file():
        missing.append(str(paths.dataset_path))
    if not paths.dataset_manifest_path.is_file():
        missing.append(str(paths.dataset_manifest_path))
    if not paths.embedding_manifest_path.is_file():
        missing.append(str(paths.embedding_manifest_path))
    if not paths.embedding_artifact_root.is_dir():
        missing.append(str(paths.embedding_artifact_root))
    if missing:
        joined = ", ".join(missing)
        raise VpiDataPackageNotInstalledError(
            "VPI data package is not installed or incomplete. "
            f"Run setup_data.py first. Missing: {joined}"
        )
