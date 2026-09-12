# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Built-in autonomy safety checks bundle (SELF-HEALING R6.4)."""

from __future__ import annotations

from intergrax.contracts.self_healing.autonomy.qualification.safety_check import AutonomySafetyCheck
from intergrax.runtime.self_healing.autonomy.qualification.checks import (
    AuditCapabilityCheck,
    DefaultPolicySafetyCheck,
    ExecutionBoundarySafetyCheck,
)


def default_autonomy_safety_checks() -> tuple[AutonomySafetyCheck, ...]:
    return (
        ExecutionBoundarySafetyCheck(),
        DefaultPolicySafetyCheck(),
        AuditCapabilityCheck(),
    )


__all__ = ["default_autonomy_safety_checks"]
