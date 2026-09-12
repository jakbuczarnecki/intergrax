# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing rollback SPI — intents only, spine executes (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.contracts.self_healing.workflow.context import SelfHealingWorkflowContext


@dataclass(frozen=True, slots=True)
class SelfHealingRollbackDirective:
    """Rollback step description — orchestrator translates via external operation spine."""

    directive_id: str
    operation_intent: str
    target_resource: str
    rationale: str

    def __post_init__(self) -> None:
        if not self.directive_id.strip():
            raise ValueError("directive_id required")
        if not self.operation_intent.strip():
            raise ValueError("operation_intent required")
        if not self.target_resource.strip():
            raise ValueError("target_resource required")
        if not self.rationale.strip():
            raise ValueError("rationale required")


@dataclass(frozen=True, slots=True)
class SelfHealingRollbackPlan:
    directives: tuple[SelfHealingRollbackDirective, ...]

    def __post_init__(self) -> None:
        if not self.directives:
            raise ValueError("directives must be non-empty")


@runtime_checkable
class SelfHealingRollbackProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def plan_rollback(
        self,
        workflow_context: SelfHealingWorkflowContext,
    ) -> SelfHealingRollbackPlan:
        """Return rollback directives — no direct infrastructure restore."""


__all__ = [
    "SelfHealingRollbackDirective",
    "SelfHealingRollbackPlan",
    "SelfHealingRollbackProvider",
]
