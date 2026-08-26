# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Direct structured inference execution backend (UE-5A)."""

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

from intergrax.contracts.execution_identity import (
    require_active_execution_identity,
    require_active_execution_id,
)
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver

OutputT = TypeVar("OutputT")


class InferenceExecutor(Generic[OutputT]):
    """Structured direct inference backend behind :class:`ExecutionBoundary`."""

    __slots__ = ("_adapter",)

    def __init__(self, adapter: LLMAdapter) -> None:
        self._adapter = adapter

    async def execute(
        self,
        request: ExecutionRequest[tuple[ChatMessage, ...], OutputT],
    ) -> OutputT:
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        del attempt_id, execution_id

        strategy = StrategyResolver().resolve(request)
        if strategy is not ExecutionStrategy.INFERENCE:
            raise RuntimeError("InferenceExecutor requires INFERENCE strategy")

        if ExecutionCapability.STREAMING in request.capabilities:
            raise RuntimeError("structured inference streaming is not implemented")

        output_type = request.output_type
        if output_type is None:
            raise RuntimeError("structured inference requires output_type")

        if not self._adapter.supports_structured_output():
            raise RuntimeError("inference adapter does not support structured output")

        def _invoke() -> OutputT:
            structured = self._adapter.generate_structured(
                request.input,
                output_type,
                run_id=run_id,
            )
            return structured.parsed

        return await asyncio.to_thread(_invoke)
