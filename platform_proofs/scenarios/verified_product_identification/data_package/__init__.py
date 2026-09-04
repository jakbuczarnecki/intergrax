"""VPI portable data package distribution."""

from platform_proofs.scenarios.verified_product_identification.data_package.config import (
    VpiDataPackageConfig,
    load_vpi_data_package_config,
)
from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageCompatibilityError,
    VpiDataPackageConfigurationError,
    VpiDataPackageError,
    VpiDataPackageNotInstalledError,
)
from platform_proofs.scenarios.verified_product_identification.data_package.install import (
    VpiDataPackageInstallResult,
    install_vpi_data_package,
)
from platform_proofs.scenarios.verified_product_identification.data_package.paths import (
    VpiInstalledDataPaths,
    assert_installed_data_present,
    resolve_installed_data_paths,
)
from platform_proofs.scenarios.verified_product_identification.data_package.validation import (
    VpiDataPackageValidationReport,
    validate_installed_vpi_data_package,
)

__all__ = [
    "VpiDataPackageCompatibilityError",
    "VpiDataPackageConfigurationError",
    "VpiDataPackageConfig",
    "VpiDataPackageError",
    "VpiDataPackageInstallResult",
    "VpiDataPackageNotInstalledError",
    "VpiDataPackageValidationReport",
    "VpiInstalledDataPaths",
    "assert_installed_data_present",
    "install_vpi_data_package",
    "load_vpi_data_package_config",
    "resolve_installed_data_paths",
    "validate_installed_vpi_data_package",
]
