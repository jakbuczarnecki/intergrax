# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from pydantic import BaseModel

from intergrax.contracts.tool_execution_outcome import (
    ToolEffectCertainty,
    ToolExecutionError,
    ToolExecutionResult,
)
from intergrax.contracts.tools.tool_invocation_wiring_resolution import (
    ToolInvocationContext,
)

if TYPE_CHECKING:
    from intergrax.tools.registry.wiring import ToolWiringContext

InModelT = TypeVar("InModelT", bound=BaseModel)
OutModelT = TypeVar("OutModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ToolExecutionRequest(Generic[InModelT]):
    """
    Runtime-owned, validated tool invocation request.

    - input MUST be an instance of the tool's input_schema (BaseModel)
    - runtime controls execution semantics via run_id + step_id
    """
    run_id: str
    step_id: str
    tool_id: str
    input: InModelT
    idempotency_key: Optional[str] = None
    declarative_hitl_invocation_scope_id: Optional[str] = None
    invocation_context: Optional[ToolInvocationContext] = None
    effective_wiring: Optional[ToolWiringContext] = None


@dataclass(frozen=True, slots=True)
class ToolModelObservation:
    """
    Model-facing serialized tool outcome for native ``role=tool`` messages.

    ``content`` is the semantically intact tool payload serialization.
    ``evidence_reference`` is the optional model-visible semantic identity
    rendered separately via :func:`format_tool_model_observation_content`.

    Distinct from diagnostic trace previews (bounded observability).
    """

    content: str
    evidence_reference: str | None = None

    @classmethod
    def from_execution_result(
        cls,
        result: ToolExecutionResult[OutModelT],
        *,
        evidence_reference: str | None = None,
    ) -> ToolModelObservation:
        if result.success:
            if result.output is None:
                raise ValueError("successful ToolExecutionResult requires output")
            return cls(
                content=result.output.model_dump_json(),
                evidence_reference=evidence_reference,
            )
        if result.error is None:
            raise ValueError("failed ToolExecutionResult requires error")
        return cls(content=result.error.error_message, evidence_reference=None)


__all__ = [
    "ToolEffectCertainty",
    "ToolExecutionError",
    "ToolExecutionRequest",
    "ToolExecutionResult",
    "ToolModelObservation",
]
