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
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileResolutionError,
    InferenceProfileResolver,
)
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.result import ExecutionResult, ExecutionStatus

OutputT = TypeVar("OutputT")


class InferenceExecutor(Generic[OutputT]):
    """Structured direct inference backend behind :class:`ExecutionBoundary`."""

    __slots__ = ("_default_adapter", "_profile_resolver")

    def __init__(
        self,
        adapter: LLMAdapter,
        *,
        profile_resolver: InferenceProfileResolver | None = None,
    ) -> None:
        self._default_adapter = adapter
        self._profile_resolver = profile_resolver

    def _select_adapter(
        self,
        request: ExecutionRequest[tuple[ChatMessage, ...], OutputT],
    ) -> LLMAdapter:
        profile_id = request.inference_profile_id
        if profile_id is None:
            return self._default_adapter
        if self._profile_resolver is None:
            raise InferenceProfileResolutionError(
                "explicit inference profile requested but no profile resolver "
                f"is configured: {profile_id!r}",
            )
        return self._profile_resolver.resolve(profile_id)

    async def execute(
        self,
        request: ExecutionRequest[tuple[ChatMessage, ...], OutputT],
    ) -> ExecutionResult[OutputT]:
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        del attempt_id, execution_id

        if ExecutionCapability.STREAMING in request.capabilities:
            raise RuntimeError("structured inference streaming is not implemented")

        output_type = request.output_type
        if output_type is None:
            raise RuntimeError("structured inference requires output_type")

        adapter = self._select_adapter(request)

        if not adapter.supports_structured_output():
            raise RuntimeError("inference adapter does not support structured output")

        def _invoke() -> OutputT:
            structured = adapter.generate_structured(
                request.input,
                output_type,
                run_id=run_id,
            )
            return structured.parsed

        parsed = await asyncio.to_thread(_invoke)
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            output=parsed,
        )
