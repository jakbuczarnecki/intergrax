# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register RAG catalog tools."""

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.rag.contracts import RagRetrieveInput, RagRetrieveOutput
from intergrax.tools.providers.rag.handler import RagRetrieveHandler
from intergrax.tools.providers.rag.ingest_contracts import RagIngestInput, RagIngestOutput
from intergrax.tools.providers.rag.ingest_handler import RagIngestHandler
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID
from intergrax.tools.providers.rag.list_collections_contracts import RagListCollectionsInput, RagListCollectionsOutput
from intergrax.tools.providers.rag.list_collections_handler import RagListCollectionsHandler
from intergrax.tools.providers.rag.lifecycle_contracts import (
    RagDeleteDocumentsInput,
    RagDeleteDocumentsOutput,
    RagDescribeCollectionInput,
    RagDescribeCollectionOutput,
)
from intergrax.tools.providers.rag.lifecycle_handler import RagDeleteDocumentsHandler, RagDescribeCollectionHandler
from intergrax.tools.providers.rag.lifecycle_service import (
    RAG_DELETE_DOCUMENTS_TOOL_ID,
    RAG_DESCRIBE_COLLECTION_TOOL_ID,
)
from intergrax.tools.providers.rag.list_collections_service import RAG_LIST_COLLECTIONS_TOOL_ID
from intergrax.tools.providers.rag.service import RAG_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_BUNDLE_ID = "rag"
RAG_TOOL_IDS: tuple[str, ...] = (
    RAG_TOOL_ID,
    RAG_INGEST_TOOL_ID,
    RAG_LIST_COLLECTIONS_TOOL_ID,
    RAG_DELETE_DOCUMENTS_TOOL_ID,
    RAG_DESCRIBE_COLLECTION_TOOL_ID,
)


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


def rag_ingest_contract() -> ToolContract:
    return ToolContract(
        tool_id=RAG_INGEST_TOOL_ID,
        name="rag.ingest_document",
        description=(
            "Load a local document through the configured document parser pipeline, chunk it, "
            "embed, and store vectors in the application index. Returns parser trace metadata."
        ),
        description_short="Ingest a file into the vector index.",
        input_schema=RagIngestInput,
        output_schema=RagIngestOutput,
        error_mapping={},
        side_effects=True,
        injects_context=False,
        category="retrieval",
        risk_level=ToolRiskLevel.MEDIUM,
        tags=("rag", "ingestion", "indexing"),
    )


def rag_list_collections_contract() -> ToolContract:
    return ToolContract(
        tool_id=RAG_LIST_COLLECTIONS_TOOL_ID,
        name="rag.list_collections",
        description=(
            "List vector index collection names available in the configured vector store. "
            "Use before ingest or retrieve when namespace selection matters."
        ),
        description_short="List vector store collections.",
        input_schema=RagListCollectionsInput,
        output_schema=RagListCollectionsOutput,
        error_mapping={},
        side_effects=False,
        category="retrieval",
        risk_level=ToolRiskLevel.LOW,
        tags=("rag", "vectorstore", "metadata"),
    )


def rag_delete_documents_contract() -> ToolContract:
    return ToolContract(
        tool_id=RAG_DELETE_DOCUMENTS_TOOL_ID,
        name="rag.delete_documents",
        description=(
            "Delete indexed vector chunks/documents by id from the configured vector store. "
            "Use after re-ingest or when removing stale knowledge base entries."
        ),
        description_short="Delete vectors by document id.",
        input_schema=RagDeleteDocumentsInput,
        output_schema=RagDeleteDocumentsOutput,
        error_mapping={},
        side_effects=True,
        category="retrieval",
        risk_level=ToolRiskLevel.HIGH,
        tags=("rag", "vectorstore", "lifecycle"),
    )


def rag_describe_collection_contract() -> ToolContract:
    return ToolContract(
        tool_id=RAG_DESCRIBE_COLLECTION_TOOL_ID,
        name="rag.describe_collection",
        description=(
            "Describe the active vector index collection: document count and available collection names."
        ),
        description_short="Vector collection stats.",
        input_schema=RagDescribeCollectionInput,
        output_schema=RagDescribeCollectionOutput,
        error_mapping={},
        side_effects=False,
        category="retrieval",
        risk_level=ToolRiskLevel.LOW,
        tags=("rag", "vectorstore", "metadata"),
    )


def register_rag_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(rag_retrieve_contract(), RagRetrieveHandler(ctx))
    registry.register(rag_ingest_contract(), RagIngestHandler(ctx))
    registry.register(rag_list_collections_contract(), RagListCollectionsHandler(ctx))
    registry.register(rag_delete_documents_contract(), RagDeleteDocumentsHandler(ctx))
    registry.register(rag_describe_collection_contract(), RagDescribeCollectionHandler(ctx))


RAG_RETRIEVE_TOOL_CONTRACT = rag_retrieve_contract()
