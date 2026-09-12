# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Enterprise validation plugin SPI (SELF-HEALING R3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.contracts.self_healing.execution.context import SelfHealingExecutionContext
    from intergrax.contracts.self_healing.observation.provider import ObservationResult
    from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext


class ValidatorCheckStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True, slots=True)
class ValidatorCheckResult:
    check_id: str
    status: ValidatorCheckStatus
    evidence_refs: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        if not self.check_id.strip():
            raise ValueError("check_id required")
        if not self.evidence_refs:
            raise ValueError("validator check requires evidence_refs")


@runtime_checkable
class SelfHealingValidator(Protocol):
    @property
    def validator_id(self) -> str: ...

    def validate(
        self,
        workflow_context: SelfHealingWorkflowContext,
        *,
        execution_context: SelfHealingExecutionContext,
        observation: ObservationResult | None = None,
    ) -> ValidatorCheckResult:
        """Evidence-only check — never returns bare validation=true."""


__all__ = ["SelfHealingValidator", "ValidatorCheckResult", "ValidatorCheckStatus"]
