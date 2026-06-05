# © Artur Czarnecki. All rights reserved.

"""Register OpenAI managed vector store catalog tools."""

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.openai_vector_store.contracts import (
    OpenAiFileSearchQueryInput,
    OpenAiFileSearchQueryOutput,
    OpenAiVectorStoreClearInput,
    OpenAiVectorStoreClearOutput,
    OpenAiVectorStoreUploadInput,
    OpenAiVectorStoreUploadOutput,
)
from intergrax.tools.providers.openai_vector_store.handlers import (
    OpenAiFileSearchQueryHandler,
    OpenAiVectorStoreClearHandler,
    OpenAiVectorStoreUploadHandler,
)
from intergrax.tools.providers.openai_vector_store.service import (
    OPENAI_FILE_SEARCH_QUERY_TOOL_ID,
    OPENAI_VECTOR_STORE_CLEAR_TOOL_ID,
    OPENAI_VECTOR_STORE_UPLOAD_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

OPENAI_VECTOR_STORE_BUNDLE_ID = "openai_vector_store"
OPENAI_VECTOR_STORE_TOOL_IDS: tuple[str, ...] = (
    OPENAI_FILE_SEARCH_QUERY_TOOL_ID,
    OPENAI_VECTOR_STORE_UPLOAD_TOOL_ID,
    OPENAI_VECTOR_STORE_CLEAR_TOOL_ID,
)


def openai_file_search_query_contract() -> ToolContract:
    return ToolContract(
        tool_id=OPENAI_FILE_SEARCH_QUERY_TOOL_ID,
        name="openai.file_search.query",
        description=(
            "Answer a question using OpenAI managed vector store retrieval via Responses API "
            "file_search. Use when documents are indexed in an OpenAI vector store (not the "
            "harness rag.retrieve index). Returns a grounded answer with strict citation rules."
        ),
        description_short="Query OpenAI hosted vector store (file_search).",
        input_schema=OpenAiFileSearchQueryInput,
        output_schema=OpenAiFileSearchQueryOutput,
        error_mapping={},
        side_effects=False,
        injects_context=True,
        category="retrieval",
        risk_level=ToolRiskLevel.LOW,
        tags=("openai", "file_search", "retrieval", "context"),
    )


def openai_vector_store_upload_contract() -> ToolContract:
    return ToolContract(
        tool_id=OPENAI_VECTOR_STORE_UPLOAD_TOOL_ID,
        name="openai.vector_store.upload",
        description=(
            "Upload local documents from a folder into an OpenAI managed vector store. "
            "Files are sent to OpenAI Files API, processed, and linked to the vector store."
        ),
        description_short="Upload folder files to OpenAI vector store.",
        input_schema=OpenAiVectorStoreUploadInput,
        output_schema=OpenAiVectorStoreUploadOutput,
        error_mapping={},
        side_effects=True,
        injects_context=False,
        category="retrieval",
        risk_level=ToolRiskLevel.MEDIUM,
        tags=("openai", "vectorstore", "ingestion"),
    )


def openai_vector_store_clear_contract() -> ToolContract:
    return ToolContract(
        tool_id=OPENAI_VECTOR_STORE_CLEAR_TOOL_ID,
        name="openai.vector_store.clear",
        description=(
            "Delete all files from an OpenAI managed vector store and remove underlying "
            "OpenAI file storage objects. Destructive — use only for explicit re-indexing."
        ),
        description_short="Clear all files from OpenAI vector store.",
        input_schema=OpenAiVectorStoreClearInput,
        output_schema=OpenAiVectorStoreClearOutput,
        error_mapping={},
        side_effects=True,
        injects_context=False,
        category="retrieval",
        risk_level=ToolRiskLevel.HIGH,
        tags=("openai", "vectorstore", "destructive"),
    )


def register_openai_vector_store_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(openai_file_search_query_contract(), OpenAiFileSearchQueryHandler(ctx))
    registry.register(openai_vector_store_upload_contract(), OpenAiVectorStoreUploadHandler(ctx))
    registry.register(openai_vector_store_clear_contract(), OpenAiVectorStoreClearHandler(ctx))


OPENAI_FILE_SEARCH_QUERY_TOOL_CONTRACT = openai_file_search_query_contract()
