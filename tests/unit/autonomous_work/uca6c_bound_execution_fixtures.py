# © Artur Czarnecki. All rights reserved.

"""Test doubles for CodeCraft bound capability execution (UCA-6C-R4)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.codecraft.bound_capability_execution import (
    CodeCraftBoundCapabilityExecutionOutcome,
    CodeCraftBoundCapabilityExecutionRequest,
    CodeCraftBoundCapabilityExecutionResult,
)
from intergrax.contracts.execution_identity import require_active_execution_id


@dataclass
class RecordingCodeCraftBoundCapabilityExecution:
    """Deterministic port — proves runtime invocation under active EE identity."""

    runtime_execution_calls: int = 0
    forced_outcome: CodeCraftBoundCapabilityExecutionOutcome = (
        CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED
    )
    forced_reason_detail: str = ""
    reject_identity_mismatch: bool = False
    _requests: list[CodeCraftBoundCapabilityExecutionRequest] = field(
        default_factory=list,
    )

    def execute(
        self,
        request: CodeCraftBoundCapabilityExecutionRequest,
    ) -> CodeCraftBoundCapabilityExecutionResult:
        self._requests.append(request)
        active = require_active_execution_id()
        if active != request.execution_id:
            if self.reject_identity_mismatch:
                return CodeCraftBoundCapabilityExecutionResult(
                    outcome=CodeCraftBoundCapabilityExecutionOutcome.REJECTED,
                    reason_detail="execution_identity_mismatch",
                )
        self.runtime_execution_calls += 1
        return CodeCraftBoundCapabilityExecutionResult(
            outcome=self.forced_outcome,
            reason_detail=self.forced_reason_detail,
        )


def recording_codecraft_execution_handler(
    *,
    side_effect_recorder: list[str] | None = None,
    forced_outcome: CodeCraftBoundCapabilityExecutionOutcome = (
        CodeCraftBoundCapabilityExecutionOutcome.SUCCEEDED
    ),
    forced_reason_detail: str = "",
) -> tuple[
    "CodeCraftQualifiedCapabilityExecutionHandler",
    RecordingCodeCraftBoundCapabilityExecution,
]:
    from intergrax.runtime.codecraft.qualified_capability_execution_handler import (
        CodeCraftQualifiedCapabilityExecutionHandler,
    )

    port = RecordingCodeCraftBoundCapabilityExecution(
        forced_outcome=forced_outcome,
        forced_reason_detail=forced_reason_detail,
    )
    handler = CodeCraftQualifiedCapabilityExecutionHandler(
        execution_port=port,
        side_effect_recorder=side_effect_recorder,
    )
    return handler, port
