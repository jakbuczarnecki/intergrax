# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical normalized tool execution outcome (shared public contract)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

OutModelT = TypeVar("OutModelT", bound=BaseModel)


class ToolEffectCertainty(str, Enum):
    """Whether a tool invocation may have crossed the external-effect boundary."""

    NOT_STARTED = "not_started"
    COMPLETED = "completed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class ToolExecutionError:
    """
    Normalized tool execution error (runtime-level).

    - error_code is already mapped by runtime enforcement (ToolContract.error_mapping)
    - error_message is safe to log/trace (no raw exception objects)
    """

    error_code: str
    error_message: str


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
    effect_certainty: ToolEffectCertainty = ToolEffectCertainty.UNCERTAIN

    @staticmethod
    def ok(output: OutModelT) -> ToolExecutionResult[OutModelT]:
        return ToolExecutionResult(
            success=True,
            output=output,
            error=None,
            effect_certainty=ToolEffectCertainty.COMPLETED,
        )

    @staticmethod
    def fail(
        code: str,
        message: str,
        *,
        effect_certainty: ToolEffectCertainty = ToolEffectCertainty.UNCERTAIN,
    ) -> ToolExecutionResult[OutModelT]:
        return ToolExecutionResult(
            success=False,
            output=None,
            error=ToolExecutionError(error_code=code, error_message=message),
            effect_certainty=effect_certainty,
        )


__all__ = [
    "ToolEffectCertainty",
    "ToolExecutionError",
    "ToolExecutionResult",
]
