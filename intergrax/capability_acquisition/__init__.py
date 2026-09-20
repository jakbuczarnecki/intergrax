# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UCA capability realization (UCA-2) and acquisition (UCA-3) coordination."""

from intergrax.capability_acquisition.acquisition_registry import (
    CapabilityAcquisitionStrategyRegistry,
)
from intergrax.capability_acquisition.acquisition_service import (
    CapabilityAcquisitionService,
)
from intergrax.capability_acquisition.availability_projection import (
    project_availability_disposition,
)
from intergrax.capability_acquisition.default_strategy_selection_policy import (
    DefaultCapabilityAcquisitionStrategySelectionPolicy,
)
from intergrax.capability_acquisition.registry import (
    CapabilityRealizationProviderRegistry,
)
from intergrax.capability_acquisition.service import CapabilityRealizationService

__all__ = [
    "CapabilityAcquisitionService",
    "CapabilityAcquisitionStrategyRegistry",
    "CapabilityRealizationProviderRegistry",
    "CapabilityRealizationService",
    "DefaultCapabilityAcquisitionStrategySelectionPolicy",
    "project_availability_disposition",
]
