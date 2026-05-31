# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register RAG catalog tools."""

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.rag.contracts import RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.providers.rag.handler import RagRetrieveHandler
from intergrax.tools.providers.rag.service import RAG_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_BUNDLE_ID = "rag"
RAG_TOOL_IDS: tuple[str, ...] = (RAG_TOOL_ID,)


def rag_retrieve_contract() -> ToolContract:
    return ToolContract(
        tool_id=RAG_TOOL_ID,
        name="rag.retrieve",
        description=(
            "Retrieve relevant document chunks from the configured vector index for a natural "
            "language query. Use when the answer should be grounded in indexed knowledge base "
            "or uploaded documents. Returns ranked snippets with scores and a compact context block."
        ),
        description_short="Search indexed documents by semantic query.",
        input_schema=RagRetrieveInput,
        output_schema=RagRetrieveOutput,
        error_mapping={},
        side_effects=False,
        injects_context=True,
        category="retrieval",
        risk_level=ToolRiskLevel.LOW,
        tags=("rag", "retrieval", "context"),
    )


def register_rag_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    handler = RagRetrieveHandler(ctx)
    registry.register(rag_retrieve_contract(), handler)
