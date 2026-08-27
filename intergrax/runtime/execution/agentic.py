# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Direct agentic execution backend (UE-6A).

RuntimeRequest input is TRANSITIONAL and must not become canonical public input.
Owner of retirement: later canonical agent-input migration / UE-9D final legacy retirement.
"""

from __future__ import annotations

from typing import Protocol

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.execution_identity import (
    require_active_execution_identity,
    require_active_execution_id,
)
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.result import ExecutionResult, ExecutionStatus
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


class AgentEnginePort(Protocol):
    """Narrow port for agentic strategy delegation to AgentEngine."""

    async def run_with_result(
        self,
        request: RuntimeRequest,
    ) -> AgentExecutionResult:
        ...


class AgentExecutor:
    """Agentic execution backend behind :class:`ExecutionBoundary`."""

    __slots__ = ("_engine",)

    def __init__(self, engine: AgentEnginePort) -> None:
        self._engine = engine

    async def execute(
        self,
        request: ExecutionRequest[RuntimeRequest, AgentExecutionResult],
    ) -> ExecutionResult[AgentExecutionResult]:
        run_id, attempt_id = require_active_execution_identity()
        execution_id = require_active_execution_id()
        del attempt_id, execution_id

        strategy = StrategyResolver().resolve(request)
        if strategy is not ExecutionStrategy.AGENTIC:
            raise RuntimeError("AgentExecutor requires AGENTIC strategy")

        if ExecutionCapability.STREAMING in request.capabilities:
            raise RuntimeError("agentic streaming is not implemented")

        if request.output_type is not AgentExecutionResult:
            raise RuntimeError("AgentExecutor requires AgentExecutionResult output_type")

        if request.input.run_id != run_id:
            raise RuntimeError(
                "agentic RuntimeRequest run_id does not match active execution"
            )

        agent_result = await self._engine.run_with_result(request.input)

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            output=agent_result,
        )
