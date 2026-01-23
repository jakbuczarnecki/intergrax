# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Protocol

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload, TraceLevel



class RuntimeRunHandle(Protocol):
    """
    Lightweight handle returned to API layer.
    """

    @property
    def run_id(self) -> str: 
        ...

    def trace_api_event(
        self,
        *,
        step: str,
        message: str,
        level: TraceLevel = TraceLevel.INFO,
        payload: Optional[DiagnosticPayload] = None,
    ) -> None: 
        ...


class RuntimeRunFactory(Protocol):
    """
    Runtime entrypoint for creating runs initiated by API.
    """

    def create_api_run(self, *, run_id: str) -> RuntimeRunHandle: 
        ...
