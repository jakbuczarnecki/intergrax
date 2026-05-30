# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.rag.contracts import RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.providers.rag.service import perform_rag_retrieve
from intergrax.tools.registry.wiring import ToolWiringContext


class RagRetrieveHandler:
    """Tool handler for ``rag.retrieve``."""

    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[RagRetrieveInput]) -> RagRetrieveOutput:
        return perform_rag_retrieve(self._ctx, request.input)
