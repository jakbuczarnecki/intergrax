# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UCA coordination errors (realization UCA-2, acquisition UCA-3)."""


class CapabilityRealizationError(Exception):
    """Base error for capability realization coordination."""


class CapabilityRealizationConfigurationError(CapabilityRealizationError):
    """Invalid provider registry or service configuration."""


class CapabilityRealizationIntegrityError(CapabilityRealizationError):
    """Provider returned inconsistent or invalid realization data."""


class CapabilityAcquisitionError(Exception):
    """Base error for capability acquisition coordination."""


class CapabilityAcquisitionConfigurationError(CapabilityAcquisitionError):
    """Invalid strategy registry or service configuration."""


class CapabilityAcquisitionIntegrityError(CapabilityAcquisitionError):
    """Strategy returned inconsistent or invalid acquisition data."""


__all__ = [
    "CapabilityAcquisitionConfigurationError",
    "CapabilityAcquisitionError",
    "CapabilityAcquisitionIntegrityError",
    "CapabilityRealizationConfigurationError",
    "CapabilityRealizationError",
    "CapabilityRealizationIntegrityError",
]
