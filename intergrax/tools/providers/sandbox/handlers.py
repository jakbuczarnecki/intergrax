# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.sandbox.contracts import SandboxExecInput, SandboxExecOutput
from intergrax.tools.providers.sandbox.service import sandbox_exec
from intergrax.tools.registry.wiring import ToolWiringContext


class SandboxExecHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[SandboxExecInput]) -> SandboxExecOutput:
        return sandbox_exec(self._ctx, request.input)
