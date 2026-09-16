# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map execution capabilities to root authorization operations (GR-2-R3)."""

from __future__ import annotations

from intergrax.contracts.root_execution_operation import RootExecutionOperation
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import (
    ExecutionStrategy,
    execution_strategy_from_capabilities,
)


def root_execution_operation_from_capabilities(
    capabilities: frozenset[ExecutionCapability],
) -> RootExecutionOperation:
    if ExecutionCapability.ORCHESTRATION in capabilities:
        return RootExecutionOperation.ROOT_ORCHESTRATION
    return RootExecutionOperation.ROOT_AGENT


def root_execution_operation_from_request(
    request: ExecutionRequest[object, object],
) -> RootExecutionOperation:
    strategy = execution_strategy_from_capabilities(request.capabilities)
    if strategy is ExecutionStrategy.INFERENCE:
        return RootExecutionOperation.ROOT_INFERENCE
    if strategy is ExecutionStrategy.ORCHESTRATION:
        return RootExecutionOperation.ROOT_ORCHESTRATION
    return RootExecutionOperation.ROOT_AGENT
