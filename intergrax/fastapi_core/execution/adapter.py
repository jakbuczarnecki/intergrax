# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Protocol

from intergrax.fastapi_core.execution.models import ExecutionRequest

class ExecutionAdapter(Protocol):
    """
    Boundary between FastAPI Core and execution world.

    Responsible ONLY for dispatching execution.
    Must NOT mutate RunStore.
    """

    async def start_execution(self, request: ExecutionRequest) -> None:
        """
        Dispatch run execution to the underlying engine/worker.

        - Must be non-blocking
        - May raise if dispatch itself fails (e.g. queue down)
        - Must NOT update run status
        """
        ...
