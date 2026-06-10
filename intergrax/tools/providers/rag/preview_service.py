# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.rag.contracts import RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.providers.rag.service import perform_rag_retrieve
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_PREVIEW_RETRIEVAL_TOOL_ID = "rag.preview_retrieval"


def rag_preview_retrieval(ctx: ToolWiringContext, params: RagRetrieveInput) -> RagRetrieveOutput:
    result = perform_rag_retrieve(ctx, params)
    if not result.used:
        return result
    preview_text = " | ".join(chunk.text[:200] for chunk in result.chunks[:5])
    return RagRetrieveOutput(
        used=True,
        chunks=result.chunks,
        citations=result.citations,
        context_text=preview_text,
        reason="preview",
        diagnostics={**result.diagnostics, "preview_mode": True},
    )
