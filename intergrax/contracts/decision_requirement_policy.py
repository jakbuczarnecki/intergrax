# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision provenance requirement policy (GR-6-R1).

Answers whether a proposed meaningful side effect must carry valid
``DecisionGovernanceMaterialRef`` before Governance may authorize it.
Does not authorize execution — ``PolicyAction`` remains Governance-only.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectKind

SCHEMA_DECISION_REQUIREMENT_CONTEXT_V1: Final = "decision_requirement_context.v1"


class DecisionRequirement(StrEnum):
    """Whether Decision governance material is mandatory for this proposal."""

    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    UNDETERMINED = "undetermined"


class DecisionRequirementContext(BaseModel):
    """Neutral execution/side-effect inputs for requirement classification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["decision_requirement_context.v1"] = (
        SCHEMA_DECISION_REQUIREMENT_CONTEXT_V1
    )
    operation_id: str = Field(min_length=1)
    resource_scope: str | None = None
    action: str = Field(min_length=1)
    kinds: tuple[MeaningfulSideEffectKind, ...] = Field(min_length=1)
    side_effect_scope_id: str = Field(min_length=1)
    side_effect_scope_digest: str | None = None
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    tenant_id: str | None = None
    principal_id: str | None = None
    resource: str | None = None
    external_target: str | None = None

    @classmethod
    def from_meaningful_side_effect(
        cls,
        *,
        operation_id: str,
        resource_scope: str | None,
        side_effect: object,
    ) -> DecisionRequirementContext:
        from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest

        if type(side_effect) is not MeaningfulSideEffectRequest:
            raise TypeError("side_effect must be MeaningfulSideEffectRequest")
        normalized_operation = operation_id.strip()
        if not normalized_operation:
            raise ValueError("operation_id must be non-empty")
        return cls(
            operation_id=normalized_operation,
            resource_scope=resource_scope,
            action=side_effect.action,
            kinds=side_effect.kinds,
            side_effect_scope_id=side_effect.side_effect_scope_id,
            side_effect_scope_digest=side_effect.side_effect_scope_digest,
            task_id=side_effect.task_id,
            run_id=side_effect.run_id,
            attempt_id=side_effect.attempt_id,
            execution_id=side_effect.execution_id,
            tenant_id=side_effect.tenant_id,
            principal_id=side_effect.principal_id,
            resource=side_effect.resource,
            external_target=side_effect.external_target,
        )


class DecisionRequirementPolicy(Protocol):
    """Plugin contract — classifies Decision provenance requirement only."""

    def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
        """Return whether Decision material is mandatory for ``context``."""
        ...


__all__ = [
    "DecisionRequirement",
    "DecisionRequirementContext",
    "DecisionRequirementPolicy",
    "SCHEMA_DECISION_REQUIREMENT_CONTEXT_V1",
]
