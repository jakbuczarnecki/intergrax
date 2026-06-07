# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.document.contracts import (
    DocumentParseInput,
    DocumentParseOutput,
    DocumentParsePreviewInput,
    DocumentParsePreviewOutput,
)
from intergrax.tools.providers.document.handlers import DocumentParseHandler, DocumentParsePreviewHandler
from intergrax.tools.providers.document.service import DOCUMENT_PARSE_PREVIEW_TOOL_ID, DOCUMENT_PARSE_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

DOCUMENT_BUNDLE_ID = "document"
DOCUMENT_TOOL_IDS: tuple[str, ...] = (DOCUMENT_PARSE_TOOL_ID, DOCUMENT_PARSE_PREVIEW_TOOL_ID)


def register_document_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=DOCUMENT_PARSE_TOOL_ID,
            name=DOCUMENT_PARSE_TOOL_ID,
            description="Parse a local document file into normalized text fragments for RAG ingestion.",
            description_short="Parse document file.",
            input_schema=DocumentParseInput,
            output_schema=DocumentParseOutput,
            error_mapping={},
            side_effects=False,
            category="document",
            risk_level=ToolRiskLevel.LOW,
            tags=("document", "parser"),
        ),
        DocumentParseHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=DOCUMENT_PARSE_PREVIEW_TOOL_ID,
            name=DOCUMENT_PARSE_PREVIEW_TOOL_ID,
            description="Parse a local document file into a bounded preview of text fragments (no ingestion).",
            description_short="Preview parsed document.",
            input_schema=DocumentParsePreviewInput,
            output_schema=DocumentParsePreviewOutput,
            error_mapping={},
            side_effects=False,
            category="document",
            risk_level=ToolRiskLevel.LOW,
            tags=("document", "parser", "preview"),
        ),
        DocumentParsePreviewHandler(ctx),
    )
