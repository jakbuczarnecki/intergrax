# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UCA-4 qualification coordination errors."""


class CapabilityQualificationError(Exception):
    """Base error for capability qualification coordination."""


class CapabilityQualificationConfigurationError(CapabilityQualificationError):
    """Invalid provider registry or service configuration."""


class CapabilityQualificationIntegrityError(CapabilityQualificationError):
    """Provider returned inconsistent or invalid qualification data."""


__all__ = [
    "CapabilityQualificationConfigurationError",
    "CapabilityQualificationError",
    "CapabilityQualificationIntegrityError",
]
