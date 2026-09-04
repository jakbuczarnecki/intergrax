"""VPI data package installation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.proof_data import (
    DataPackageCache,
    DataPackageInstaller,
    DataPackageInstallRequest,
    DataPackageInstallReport,
    HttpDataPackageTransport,
    LocalFileDataPackageTransport,
    load_proof_data_package_descriptor,
)

from platform_proofs.scenarios.verified_product_identification.data_package.config import (
    VpiDataPackageConfig,
    load_vpi_data_package_config,
)
from platform_proofs.scenarios.verified_product_identification.data_package.validation import (
    VpiDataPackageValidationReport,
    validate_installed_vpi_data_package,
)


@dataclass(frozen=True, slots=True)
class VpiDataPackageInstallResult:
    install_report: DataPackageInstallReport
    validation_report: VpiDataPackageValidationReport


def install_vpi_data_package(
    config: VpiDataPackageConfig | None = None,
    *,
    local_mirror_root: Path | None = None,
) -> VpiDataPackageInstallResult:
    resolved_config = config or load_vpi_data_package_config()
    descriptor = load_proof_data_package_descriptor(resolved_config.descriptor_path)
    cache = DataPackageCache(resolved_config.cache_dir)
    transport = _select_transport(local_mirror_root)
    base_uri = _resolve_base_uri(resolved_config.base_uri, local_mirror_root)

    installer = DataPackageInstaller()
    install_report = installer.install(
        DataPackageInstallRequest(
            descriptor=descriptor,
            install_root=resolved_config.install_dir,
            cache=cache,
            transport=transport,
            base_uri=base_uri,
        )
    )
    validation_report = validate_installed_vpi_data_package(
        resolved_config.install_dir,
        descriptor_path=resolved_config.descriptor_path,
    )
    return VpiDataPackageInstallResult(
        install_report=install_report,
        validation_report=validation_report,
    )


def _select_transport(local_mirror_root: Path | None):
    if local_mirror_root is not None:
        return LocalFileDataPackageTransport()
    return HttpDataPackageTransport()


def _resolve_base_uri(
    configured_base_uri: str | None,
    local_mirror_root: Path | None,
) -> str | None:
    if local_mirror_root is not None:
        normalized = local_mirror_root.resolve().as_uri()
        if not normalized.endswith("/"):
            normalized = f"{normalized}/"
        return normalized
    return configured_base_uri
