"""Unit tests for VPI data package integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.proof_data import LocalFileDataPackageTransport, load_proof_data_package_descriptor
from intergrax.proof_data.cache import DataPackageCache
from intergrax.proof_data.installer import DataPackageInstaller, DataPackageInstallRequest
from platform_proofs.scenarios.verified_product_identification.data_package.config import (
    VpiDataPackageConfig,
)
from platform_proofs.scenarios.verified_product_identification.data_package.install import (
    install_vpi_data_package,
)
from platform_proofs.scenarios.verified_product_identification.data_package.validation import (
    validate_installed_vpi_data_package,
)


def _fixture_root() -> Path:
    return (
        Path(__file__).resolve().parents[5]
        / "platform_proofs"
        / "scenarios"
        / "verified_product_identification"
        / "data_package"
        / "fixtures"
        / "tiny_v1"
    )


def test_vpi_fixture_install_and_validate(tmp_path: Path) -> None:
    fixture = _fixture_root()
    config = VpiDataPackageConfig(
        descriptor_path=fixture / "package.json",
        install_dir=tmp_path / "installed",
        cache_dir=tmp_path / "cache",
        base_uri=None,
        package_version="tiny-1",
    )
    result = install_vpi_data_package(config, local_mirror_root=fixture)
    assert result.install_report.files_total >= 1
    validation = validate_installed_vpi_data_package(
        config.install_dir,
        descriptor_path=config.descriptor_path,
    )
    assert validation.dataset_record_count == 3_770_377
    assert validation.package_id == "verified-product-identification"


def test_vpi_validation_rejects_bad_dataset_checksum(tmp_path: Path) -> None:
    fixture = _fixture_root()
    config = VpiDataPackageConfig(
        descriptor_path=fixture / "package.json",
        install_dir=tmp_path / "installed",
        cache_dir=tmp_path / "cache",
        base_uri=None,
        package_version="tiny-1",
    )
    install_vpi_data_package(config, local_mirror_root=fixture)
    manifest_path = tmp_path / "installed" / "dataset" / "manifest.json"
    manifest_path.write_text('{"builder_version":"x"}', encoding="utf-8")
    with pytest.raises(Exception):
        validate_installed_vpi_data_package(config.install_dir)
