# © Artur Czarnecki. All rights reserved.

"""Production durability requirements for execution-scoped skill pinning (SKILL-X Batch C)."""

from __future__ import annotations

from typing import Protocol

from intergrax.skills.resolver import SkillResolutionError

DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG = (
    "durable skill execution pinning store required for production execution recovery"
)


class DurableSkillPinningCapability(Protocol):
    @property
    def is_durable(self) -> bool:
        """Whether pinning state survives process restart."""


def validate_skill_execution_pinning_durability_for_composition(
    *,
    production_mode: bool,
    pinning_store: DurableSkillPinningCapability,
) -> None:
    if not production_mode:
        return
    if not pinning_store.is_durable:
        raise SkillResolutionError(DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG)


__all__ = [
    "DURABLE_SKILL_EXECUTION_PINNING_REQUIRED_MSG",
    "DurableSkillPinningCapability",
    "validate_skill_execution_pinning_durability_for_composition",
]
