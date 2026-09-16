# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic execution strategy resolution (UE-3A)."""

from __future__ import annotations

from enum import Enum
from typing import TypeVar

from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class ExecutionStrategy(str, Enum):
    """Canonical execution strategy categories selected from explicit capabilities."""

    INFERENCE = "inference"
    AGENTIC = "agentic"
    ORCHESTRATION = "orchestration"


def execution_strategy_from_capabilities(
    capabilities: frozenset[ExecutionCapability],
) -> ExecutionStrategy:
    """Deterministic capability-to-strategy mapping shared by router and metadata surfaces."""
    if ExecutionCapability.ORCHESTRATION in capabilities:
        return ExecutionStrategy.ORCHESTRATION
    if (
        ExecutionCapability.AGENT in capabilities
        or ExecutionCapability.TOOLS in capabilities
    ):
        return ExecutionStrategy.AGENTIC
    return ExecutionStrategy.INFERENCE


class StrategyResolver:
    """Stateless resolver mapping ExecutionRequest capabilities to ExecutionStrategy."""

    __slots__ = ()

    def resolve(
        self,
        request: ExecutionRequest[InputT, OutputT],
    ) -> ExecutionStrategy:
        return execution_strategy_from_capabilities(request.capabilities)
