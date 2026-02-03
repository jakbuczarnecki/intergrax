# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Protocol, runtime_checkable

from intergrax.fastapi_core.execution.models import ExecutionRequest


class ExecutionWorker(Protocol):
    """
    Pure execution contract.

    Worker:
    - performs execution logic
    - may raise exception
    - does NOT manage lifecycle
    """

    def execute(self, request: ExecutionRequest) -> None: ...


@runtime_checkable
class CancellableExecutionWorker(ExecutionWorker, Protocol):
    def cancel(self, run_id: str) -> None: ...