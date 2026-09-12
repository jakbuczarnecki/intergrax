"""Explicit provisioning failure taxonomy — environment and qualification-setup only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.phases import (
    ProvisioningPhase,
)


class ProvisioningFailureCode(str, Enum):
    """Failure classes aligned with ERL-QUAL-004 data provisioning architecture § 8."""

    INVALID_DATASET = "INVALID_DATASET"
    MISSING_SCENARIO_VARIANT = "MISSING_SCENARIO_VARIANT"
    PROVISIONING_UNAVAILABLE = "PROVISIONING_UNAVAILABLE"
    INCOMPLETE_PROVISIONING = "INCOMPLETE_PROVISIONING"
    INCONSISTENT_ENVIRONMENT = "INCONSISTENT_ENVIRONMENT"
    CLEANUP_FAILURE = "CLEANUP_FAILURE"


@dataclass(frozen=True, slots=True)
class ProvisioningFailure:
    """Typed failure returned in phase outcomes — not raised as the primary control flow."""

    code: ProvisioningFailureCode
    phase: ProvisioningPhase
    message: str
