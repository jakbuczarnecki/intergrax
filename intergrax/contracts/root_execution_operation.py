# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform-owned root execution authorization operation identity (GR-2-R3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

WORKER_ROOT_EXECUTION_OPERATION: Final = "worker.root_execution.dispatch"


class RootExecutionOperation(StrEnum):
    """Policy operation keys for root execution admission — not execution capabilities."""

    ROOT_INFERENCE = "root.execution.inference"
    ROOT_AGENT = "root.execution.agent"
    ROOT_ORCHESTRATION = "root.execution.orchestration"
    ROOT_WORKER_DISPATCH = WORKER_ROOT_EXECUTION_OPERATION

    def policy_operation(self) -> str:
        return str(self.value)


def normalize_root_execution_policy_operation(
    operation: RootExecutionOperation | str,
) -> str:
    if isinstance(operation, RootExecutionOperation):
        return operation.policy_operation()
    normalized = operation.strip()
    if not normalized:
        raise ValueError("root execution operation must be non-empty")
    return normalized
