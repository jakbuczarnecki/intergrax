# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intergrax.logging import IntergraxLogging
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.run_factory.contracts import (
    RuntimeRunFactory,
    RuntimeRunHandle,
)
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload, TraceComponent, TraceLevel


@dataclass
class _DryRunHandle(RuntimeRunHandle):
    """
    Runtime-owned handle exposed to API.
    API must not access RuntimeState directly.
    """

    _state: RuntimeState

    @property
    def run_id(self) -> str:
        return self._state.run_id

    def trace_api_event(
        self,
        *,
        step: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        payload: Optional[DiagnosticPayload] = None,
    ) -> None:
        IntergraxLogging.set_state(self._state)

        self._state.trace_event(
            component=TraceComponent.RUNTIME,
            step=step,
            message=message,
            level=level,
            payload=payload,
        )


class DryRunRuntimeRunFactory(RuntimeRunFactory):
    """
    Runtime entrypoint for API-originated dry runs.

    Guarantees:
    - Builds a fully valid RuntimeState
    - Emits lifecycle trace events
    - Does NOT execute any runtime steps
    """

    def __init__(
        self,
        *,
        runtime_context: RuntimeContext,
        system_user_id: str = "__api__",
        system_session_id: str = "__api__",
    ) -> None:
        self._context = runtime_context
        self._system_user_id = system_user_id
        self._system_session_id = system_session_id

    def create_api_run(self, *, run_id: str) -> RuntimeRunHandle:
        # Synthetic, explicit API-originated request
        request = RuntimeRequest(
            user_id=self._system_user_id,
            session_id=self._system_session_id,
            message="[API dry-run]",
            metadata={"origin": "api", "mode": "dry-run"},
        )

        state = RuntimeState(
            context=self._context,
            request=request,
            run_id=run_id,
        )

        state.trace_event(
            component=TraceComponent.RUNTIME,
            step="create",
            message="Run created via API (dry-run)",
            level=TraceLevel.INFO,
        )

        return _DryRunHandle(_state=state)
