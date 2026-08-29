# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical execution strategy routing (UE-9DR1)."""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ResultT = TypeVar("ResultT")


class OrchestrationRouterDelegate(Protocol):
    """Orchestration backend invoked by :class:`StrategyExecutionRouter`."""

    async def execute(
        self,
        request: ExecutionRequest[InputT, OutputT],
    ) -> ResultT:
        ...


class StrategyExecutionRouter(Generic[InputT, OutputT, ResultT]):
    """
    Typed execution delegate that resolves strategy once and routes to a backend.

    Owns the sole production :class:`StrategyResolver` invocation for a wired stack.
    """

    __slots__ = (
        "_resolver",
        "_inference_executor",
        "_agent_executor",
        "_orchestration_executor",
    )

    def __init__(
        self,
        *,
        resolver: StrategyResolver | None = None,
        inference_executor: InferenceExecutor[OutputT] | None = None,
        agent_executor: AgentExecutor | None = None,
        orchestration_executor: OrchestrationRouterDelegate | None = None,
    ) -> None:
        self._resolver = resolver or StrategyResolver()
        self._inference_executor = inference_executor
        self._agent_executor = agent_executor
        self._orchestration_executor = orchestration_executor

    async def execute(
        self,
        request: ExecutionRequest[InputT, OutputT],
    ) -> ResultT:
        strategy = self._resolver.resolve(request)

        if strategy is ExecutionStrategy.INFERENCE:
            if self._inference_executor is None:
                raise RuntimeError(
                    "INFERENCE strategy is not configured for this execution router"
                )
            return await self._inference_executor.execute(request)

        if strategy is ExecutionStrategy.AGENTIC:
            if self._agent_executor is None:
                raise RuntimeError(
                    "AGENTIC strategy is not configured for this execution router"
                )
            return await self._agent_executor.execute(request)

        if self._orchestration_executor is None:
            raise RuntimeError(
                "ORCHESTRATION strategy is not configured for this execution router"
            )
        return await self._orchestration_executor.execute(request)
