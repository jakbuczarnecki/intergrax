# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X5 / X5A external provider qualification support (harness + inventory)."""

from testing_support.obs_diag_provider_qualification.descriptor import (
    ObsDiagProviderClass,
    ObsDiagProviderDomain,
    ObsDiagProviderQualificationDescriptor,
    ObsDiagProviderSupportStatus,
)
from testing_support.obs_diag_provider_qualification.discovery import (
    DiscoveredObsDiagProvider,
    discover_obs_diag_provider_surfaces,
)
from testing_support.obs_diag_provider_qualification.inventory import (
    OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS,
    OBS_DIAG_X5_PROVIDER_INVENTORY,
    build_obs_diag_external_classifications,
)
from testing_support.obs_diag_provider_qualification.reconciliation import (
    obs_diag_anti_drift_delta,
    obs_diag_qualified_external_without_proof,
    obs_diag_telemetry_vendor_falsely_qualified,
)

__all__ = [
    "DiscoveredObsDiagProvider",
    "OBS_DIAG_EXTERNAL_PROVIDER_CLASSIFICATIONS",
    "OBS_DIAG_X5_PROVIDER_INVENTORY",
    "ObsDiagProviderClass",
    "ObsDiagProviderDomain",
    "ObsDiagProviderQualificationDescriptor",
    "ObsDiagProviderSupportStatus",
    "build_obs_diag_external_classifications",
    "discover_obs_diag_provider_surfaces",
    "obs_diag_anti_drift_delta",
    "obs_diag_qualified_external_without_proof",
    "obs_diag_telemetry_vendor_falsely_qualified",
]
