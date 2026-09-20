# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Realization coordination errors (UCA-2)."""


class CapabilityRealizationError(Exception):
    """Base error for capability realization coordination."""


class CapabilityRealizationConfigurationError(CapabilityRealizationError):
    """Invalid provider registry or service configuration."""


class CapabilityRealizationIntegrityError(CapabilityRealizationError):
    """Provider returned inconsistent or invalid realization data."""


__all__ = [
    "CapabilityRealizationConfigurationError",
    "CapabilityRealizationError",
    "CapabilityRealizationIntegrityError",
]
