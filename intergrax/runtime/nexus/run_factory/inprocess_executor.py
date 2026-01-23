# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Awaitable, Callable

from intergrax.runtime.nexus.run_factory.contracts import RuntimeRunHandle
from intergrax.runtime.nexus.run_factory.executor_contracts import RuntimeRunExecutor
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload, TraceLevel


class InProcessRuntimeRunExecutor(RuntimeRunExecutor):
    """
    Executes runtime runs in-process using an injected execution callable.
    """

    def __init__(
        self,
        *,
        execute_callable: Callable[[RuntimeRunHandle], Awaitable[None]],
    ) -> None:
        self._execute_callable = execute_callable

    async def execute(self, *, handle: RuntimeRunHandle) -> None:
        handle.trace_api_event(
            step="execute.start",
            message="Execution started via API",
            level=TraceLevel.INFO,
        )

        try:
            await self._execute_callable(handle)

            handle.trace_api_event(
                step="execute.complete",
                message="Execution completed",
                level=TraceLevel.INFO,
            )

        except Exception:
            # Do NOT invent DiagnosticPayload constructors.
            # Exception details are handled by logging + global error handlers.
            handle.trace_api_event(
                step="execute.error",
                message="Execution failed",
                level=TraceLevel.ERROR,
            )
            raise

