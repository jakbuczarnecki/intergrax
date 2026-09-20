# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X5 external provider qualification support (harness + inventory)."""

from testing_support.obs_diag_provider_qualification.descriptor import (
    ObsDiagProviderQualificationDescriptor,
    ObsDiagProviderSupportStatus,
)
from testing_support.obs_diag_provider_qualification.inventory import (
    OBS_DIAG_X5_PROVIDER_INVENTORY,
)

__all__ = [
    "OBS_DIAG_X5_PROVIDER_INVENTORY",
    "ObsDiagProviderQualificationDescriptor",
    "ObsDiagProviderSupportStatus",
]
