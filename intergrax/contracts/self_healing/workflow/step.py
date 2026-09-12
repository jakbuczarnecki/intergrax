# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing workflow step contract (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelfHealingStep:
    """Declarative healing step — orchestrator delegates execution to the external spine."""

    step_id: str
    operation_intent: str
    required_capability: str
    sequence_number: int
    validation_requirements: tuple[str, ...]
    rollback_reference: str | None

    def __post_init__(self) -> None:
        if not self.step_id.strip():
            raise ValueError("step_id required")
        if not self.operation_intent.strip():
            raise ValueError("operation_intent required")
        if not self.required_capability.strip():
            raise ValueError("required_capability required")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")


__all__ = ["SelfHealingStep"]
