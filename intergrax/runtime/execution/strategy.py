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


class StrategyResolver:
    """Stateless resolver mapping ExecutionRequest capabilities to ExecutionStrategy."""

    __slots__ = ()

    def resolve(
        self,
        request: ExecutionRequest[InputT, OutputT],
    ) -> ExecutionStrategy:
        capabilities = request.capabilities

        if ExecutionCapability.ORCHESTRATION in capabilities:
            return ExecutionStrategy.ORCHESTRATION

        if (
            ExecutionCapability.AGENT in capabilities
            or ExecutionCapability.TOOLS in capabilities
        ):
            return ExecutionStrategy.AGENTIC

        return ExecutionStrategy.INFERENCE
