# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.notify.contracts import NotifySendInput, NotifySendOutput
from intergrax.tools.providers.notify.service import notify_send
from intergrax.tools.registry.wiring import ToolWiringContext


class NotifySendHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[NotifySendInput]) -> NotifySendOutput:
        return notify_send(self._ctx, request.input)
