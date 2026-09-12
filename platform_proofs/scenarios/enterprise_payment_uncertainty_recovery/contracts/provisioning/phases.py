"""Provisioning lifecycle phase and outcome status enumerations."""

from __future__ import annotations

from enum import Enum


class ProvisioningPhase(str, Enum):
    """Ordered lifecycle phases owned by the provisioning component."""

    PREPARE = "PREPARE"
    PROVISION = "PROVISION"
    STATE_AVAILABILITY = "STATE_AVAILABILITY"
    CLEANUP = "CLEANUP"


class ProvisioningOutcomeStatus(str, Enum):
    """Explicit success or failure for each phase."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
