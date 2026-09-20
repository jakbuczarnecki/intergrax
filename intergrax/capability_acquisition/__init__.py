# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UCA capability realization coordination (UCA-2) — not domain lifecycle owner."""

from intergrax.capability_acquisition.availability_projection import (
    project_availability_disposition,
)
from intergrax.capability_acquisition.registry import (
    CapabilityRealizationProviderRegistry,
)
from intergrax.capability_acquisition.service import CapabilityRealizationService

__all__ = [
    "CapabilityRealizationProviderRegistry",
    "CapabilityRealizationService",
    "project_availability_disposition",
]
