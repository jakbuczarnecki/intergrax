# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

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


@dataclass(frozen=True, slots=True)
class ToolExecutionError:
    """
    Normalized tool execution error (runtime-level).

    - error_code is already mapped by runtime enforcement (ToolContract.error_mapping)
    - error_message is safe to log/trace (no raw exception objects)
    """
    error_code: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ToolModelObservation:
    """
    Model-facing serialized tool outcome for native ``role=tool`` messages.

    Distinct from diagnostic trace previews (bounded observability).
    """

    content: str

    @classmethod
    def from_execution_result(
        cls,
        result: "ToolExecutionResult[OutModelT]",
    ) -> "ToolModelObservation":
        if result.success:
            if result.output is None:
                raise ValueError("successful ToolExecutionResult requires output")
            return cls(content=result.output.model_dump_json())
        if result.error is None:
            raise ValueError("failed ToolExecutionResult requires error")
        return cls(content=result.error.error_message)


@dataclass(frozen=True)
class ToolExecutionResult(Generic[OutModelT]):
    """
    Normalized tool execution outcome.

    Exactly one of:
      - output (success=True)
      - error  (success=False)

    CONTRACT:
    If success=True → output MUST be instance of ToolContract.output_schema.
    Runtime relies on this guarantee.
    """
    success: bool
    output: Optional[OutModelT]
    error: Optional[ToolExecutionError]

    @staticmethod
    def ok(output: OutModelT) -> "ToolExecutionResult[OutModelT]":
        return ToolExecutionResult(success=True, output=output, error=None)

    @staticmethod
    def fail(code: str, message: str) -> "ToolExecutionResult[OutModelT]":
        return ToolExecutionResult(
            success=False,
            output=None,
            error=ToolExecutionError(error_code=code, error_message=message),
        )

