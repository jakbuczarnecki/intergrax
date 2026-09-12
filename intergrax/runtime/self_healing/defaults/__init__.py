# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Platform default self-healing strategies — plugins, not authority (SELF-HEALING R1)."""

from intergrax.runtime.self_healing.defaults.capacity_protection import CapacityProtectionStrategy
from intergrax.runtime.self_healing.defaults.dependency_isolation import (
    ExternalDependencyIsolationStrategy,
)
from intergrax.runtime.self_healing.defaults.generic_retry import GenericRetryAdjustmentStrategy
from intergrax.runtime.self_healing.defaults.human_escalation import HumanEscalationStrategy


def platform_default_strategies() -> tuple[
    GenericRetryAdjustmentStrategy,
    CapacityProtectionStrategy,
    ExternalDependencyIsolationStrategy,
    HumanEscalationStrategy,
]:
    return (
        GenericRetryAdjustmentStrategy(),
        CapacityProtectionStrategy(),
        ExternalDependencyIsolationStrategy(),
        HumanEscalationStrategy(),
    )


__all__ = [
    "CapacityProtectionStrategy",
    "ExternalDependencyIsolationStrategy",
    "GenericRetryAdjustmentStrategy",
    "HumanEscalationStrategy",
    "platform_default_strategies",
]
