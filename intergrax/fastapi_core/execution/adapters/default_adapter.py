# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.fastapi_core.execution.adapters.adapter import ExecutionAdapter
from intergrax.fastapi_core.execution.models import ExecutionRequest


class DefaultExecutionAdapter(ExecutionAdapter):
    """
    Default no-op execution adapter.

    - Used when no real execution engine is configured
    - Dispatches nothing
    - Does NOT mutate RunStore
    """

    async def start_execution(self, request: ExecutionRequest) -> None:
        # Intentionally no-op
        return
    
    def shutdown(self, wait: bool = True) -> None:
        return
