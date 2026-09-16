# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ME-12 machine capability acquisition errors."""


class MachineCapabilityAcquisitionError(Exception):
    """Base error for machine acquisition boundary."""


class MachineCapabilityAcquisitionSelectionError(MachineCapabilityAcquisitionError):
    """Selection is not admissible against governed recommendations."""


class MachineCapabilityAcquisitionPolicyError(MachineCapabilityAcquisitionError):
    """Custom acquisition policy attempted to widen recommendations."""


__all__ = [
    "MachineCapabilityAcquisitionError",
    "MachineCapabilityAcquisitionPolicyError",
    "MachineCapabilityAcquisitionSelectionError",
]
