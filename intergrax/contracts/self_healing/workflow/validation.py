# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing validation SPI (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext


class ValidationStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    status: ValidationStatus
    confidence: float
    evidence_refs: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.explanation.strip():
            raise ValueError("explanation required")
        if not self.evidence_refs:
            raise ValueError("validation must reference evidence")


@runtime_checkable
class SelfHealingValidationProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def validate(self, workflow_context: SelfHealingWorkflowContext) -> ValidationResult:
        """Evidence-based validation — no subjective success claims."""
        ...


__all__ = [
    "SelfHealingValidationProvider",
    "ValidationResult",
    "ValidationStatus",
]
