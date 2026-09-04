"""VPI data package configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageConfigurationError,
)

VPI_DATA_PACKAGE_ENV_PREFIX = "VPI_DATA_PACKAGE"
DEFAULT_PACKAGE_ID = "verified-product-identification"
DEFAULT_PACKAGE_VERSION = "1.0.0"
PACKAGE_DESCRIPTOR_FILENAME = "package.json"


def _scenario_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_descriptor_path() -> Path:
    return (
        _scenario_root()
        / "data_package"
        / "v1"
        / PACKAGE_DESCRIPTOR_FILENAME
    )


def default_install_dir() -> Path:
    return _scenario_root() / "data_package" / "installed"


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "intergrax" / "proof-data"


@dataclass(frozen=True, slots=True)
class VpiDataPackageConfig:
    descriptor_path: Path
    install_dir: Path
    cache_dir: Path
    base_uri: str | None
    package_version: str

    def __post_init__(self) -> None:
        if not self.package_version.strip():
            raise VpiDataPackageConfigurationError("package_version must be non-empty")


def load_vpi_data_package_config() -> VpiDataPackageConfig:
    prefix = VPI_DATA_PACKAGE_ENV_PREFIX
    descriptor_raw = os.getenv(f"{prefix}_DESCRIPTOR_PATH", "").strip()
    descriptor_path = Path(descriptor_raw) if descriptor_raw else default_descriptor_path()

    install_raw = os.getenv(f"{prefix}_INSTALL_DIR", "").strip()
    install_dir = Path(install_raw) if install_raw else default_install_dir()

    cache_raw = os.getenv(f"{prefix}_CACHE_DIR", "").strip()
    cache_dir = Path(cache_raw) if cache_raw else default_cache_dir()

    base_uri = os.getenv(f"{prefix}_BASE_URL", "").strip() or None
    package_version = os.getenv(
        f"{prefix}_VERSION",
        DEFAULT_PACKAGE_VERSION,
    ).strip()

    return VpiDataPackageConfig(
        descriptor_path=descriptor_path,
        install_dir=install_dir,
        cache_dir=cache_dir,
        base_uri=base_uri,
        package_version=package_version,
    )
